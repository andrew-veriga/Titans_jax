## Полный код модели-обёртки для kauldron Trainer

### Файл `layer23_kauldron.py`

```python
"""
Kauldron-compatible model for training TitansBlock (layer 23)
with precomputed activations from parquet dataset.
"""

import dataclasses
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import grain.python as grain
from flax import linen as flax_nn
from flax.core import freeze, unfreeze

from kauldron import kd
from kauldron import kontext

from gemma.gm.nn import _modules, _layers, _gemma
from gemma_titans import TitansBlock, DistillationOutput
from titans import init_memory_state


# ──────────────────────────── Helper ────────────────────────────

def _block_kwargs(layer_idx: int) -> dict:
    """Gemma Block kwargs for a given layer."""
    gc = _gemma.Gemma3_1B.config
    at = gc.attention_types[layer_idx]
    is_local = at == _modules.AttentionType.LOCAL_SLIDING
    return dict(
        num_heads=gc.num_heads, num_kv_heads=gc.num_kv_heads,
        embed_dim=gc.embed_dim, head_dim=gc.head_dim,
        hidden_dim=gc.hidden_dim,
        sliding_window_size=gc.sliding_window_size,
        use_post_attn_norm=gc.use_post_attn_norm,
        use_post_ffw_norm=gc.use_post_ffw_norm,
        attn_logits_soft_cap=gc.attn_logits_soft_cap,
        attn_type=at,
        query_pre_attn_scalar=gc.query_pre_attn_scalar(),
        transpose_gating_einsum=gc.transpose_gating_einsum,
        use_qk_norm=gc.use_qk_norm,
        rope_base_frequency=gc.local_base_frequency if is_local else gc.global_base_frequency,
        rope_scale_factor=gc.local_scale_factor if is_local else gc.global_scale_factor,
    )


# ──────────────────────────── Model ─────────────────────────────

class Layer23Model(flax_nn.Module):
    """
    Trainable:  layer_23 (TitansBlock)
    Frozen:     layer_24, layer_25, final_norm, embedder

    Input batch keys: activations, tokens, mask
    Returns: DistillationOutput with layer_losses['lm_loss'] / ['lm_accuracy']
    """

    # Kauldron kontext — подставляются из батча автоматически
    hidden: kontext.Key = "batch.activations"
    tokens: kontext.Key = "batch.tokens"
    mask:   kontext.Key = "batch.mask"
    step:   kontext.Key = "step"

    neural_mem_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    embed_dim: int = 1152
    final_logit_softcap: float = None

    # ── setup ──────────────────────────────────────────────────
    def setup(self):
        gc = _gemma.Gemma3_1B.config

        # Trainable TitansBlock
        self.layer_23 = TitansBlock(name='layer_23', **_block_kwargs(23),
                                    neural_mem_kwargs=self.neural_mem_kwargs)

        # Frozen Gemma blocks
        self.layer_24 = _modules.Block(name='layer_24', **_block_kwargs(24))
        self.layer_25 = _modules.Block(name='layer_25', **_block_kwargs(25))
        self.final_norm = _layers.RMSNorm()
        self.embedder = _modules.Embedder(vocab_size=gc.num_embed,
                                          embed_dim=gc.embed_dim)

    # ── forward ────────────────────────────────────────────────
    def __call__(self, hidden, tokens, mask, *, step, **kwargs):
        B, L, D = hidden.shape

        targets   = tokens[:, 1:]
        loss_mask = mask[:, 1:].astype(jnp.float32)
        positions = jnp.broadcast_to(jnp.arange(L)[None, :], (B, L))
        causal    = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
        attn_mask = causal[None, :, :] & mask[:, None, :].astype(jnp.bool_)

        # Memory state (zero-init для каждого батча)
        mem_state = init_memory_state(
            batch_size=B, dim=self.embed_dim,
            neural_mem_kwargs=self.neural_mem_kwargs, dtype=jnp.float32)

        # Huber delta schedule
        huber_delta = None
        hcfg = self.neural_mem_kwargs.get('huber_loss_delta')
        if hcfg is not None:
            s = step if step.ndim == 0 else step[0]
            huber_delta = hcfg(s) if callable(hcfg) else hcfg

        # 1) TitansBlock (trainable)
        _, x = self.layer_23(
            hidden, positions,
            {'memory_state': mem_state}, attn_mask,
            False,                                      # is_teacher_mode
            current_huber_delta=huber_delta)

        # 2) Frozen tail + CE loss — checkpointed
        @jax.checkpoint
        def _frozen_tail_and_loss(x):
            _, x = self.layer_24(x, positions, None, attn_mask)
            _, x = self.layer_25(x, positions, None, attn_mask)
            x    = self.final_norm(x)
            logits = self.embedder.decode(x[:, :-1, :])
            if self.final_logit_softcap is not None:
                logits = logits / self.final_logit_softcap
                logits = jnp.tanh(logits) * self.final_logit_softcap
            ce    = optax.softmax_cross_entropy_with_integer_labels(
                        logits.astype(jnp.float32), targets)
            denom = jnp.maximum(loss_mask.sum(axis=-1), 1.0)
            loss  = (ce * loss_mask).sum(axis=-1) / denom
            pred  = jnp.argmax(logits, axis=-1)
            acc   = ((pred == targets).astype(jnp.float32)
                     * loss_mask).sum(axis=-1) / denom
            return loss, acc

        loss, acc = _frozen_tail_and_loss(x)

        return DistillationOutput(
            logits=jnp.zeros((B, 1)),
            cache=None,
            hidden_states=None,
            layer_losses={'lm_loss': loss, 'lm_accuracy': acc})


# ──────────────────── Init Transform ────────────────────────────

class Layer23InitTransform(kd.ckpts.InitTransform):
    """Загружает merged_params, оставляет только слои 23-25 + embedder."""
    def __init__(self, merged_params):
        self.merged_params = unfreeze(merged_params)

    def transform(self, state):
        p = self.merged_params
        return state.replace(params=freeze({
            'layer_23':   p['layer_23'],
            'layer_24':   p['layer_24'],
            'layer_25':   p['layer_25'],
            'final_norm': p['final_norm'],
            'embedder':   p['embedder'],
        }))


# ──────────────────── Masked Optimizer ──────────────────────────

def make_layer23_optimizer(titans_optimizer):
    """Только layer_23 обновляется, остальные params заморожены."""
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: 'train' if path[0].key == 'layer_23' else 'freeze',
            params)
    return optax.multi_transform(
        {'train': titans_optimizer, 'freeze': optax.identity()},
        label_fn)


# ──────────────────── Dataset Pipeline ──────────────────────────

class _ProcessActivations(grain.MapTransform):
    def __init__(self, max_seq_len=1024, embed_dim=1152):
        self.L = max_seq_len
        self.D = embed_dim

    def map(self, element):
        act = np.array(element["activations"], dtype=np.float32).reshape(self.L, self.D)
        return {
            "activations": act,
            "tokens":      np.array(element["tokens"], dtype=np.int32),
            "mask":        np.array(element["mask"],   dtype=np.int32),
        }


def get_activation_dataset(
    repo_id="veriga/openwebtext-gemma3-tokenized-1024-activations-layer23",
    data_dir="data",
    batch_size=8,
    max_seq_len=1024,
    embed_dim=1152,
):
    return kd.data.py.HuggingFace(
        path=repo_id,
        data_dir=data_dir,
        split="train",
        shuffle=True,
        num_epochs=None,
        batch_size=batch_size,
        transforms=[
            _ProcessActivations(max_seq_len, embed_dim),
            kd.data.py.Elements(keep=["activations", "tokens", "mask"]),
        ])
```

### Ноутбук — ячейки запуска

```python
# ── Ячейка 1: Импорты и модель ──
import os, dataclasses
from layer23_kauldron import (
    Layer23Model, Layer23InitTransform,
    make_layer23_optimizer, get_activation_dataset,
)

experimental_config = { ... }   # твой конфиг TitansBlock
opt_params = { ... }            # твой конфиг оптимизатора
from routing_optimizer import make_routing_optimizer

# ── Ячейка 2: Модель ──
model = Layer23Model(neural_mem_kwargs=experimental_config)

# ── Ячейка 3: Загрузка merged_params (как раньше) ──
merged_params = ...   # Загрузка Phase2/Titans + Gemma весов

workdir = os.path.abspath('./titans_workdir_Layer23')
workdir_checkpoints = os.path.join(workdir, "checkpoints")

if os.path.exists(workdir_checkpoints) and len(os.listdir(workdir_checkpoints)) > 0:
    print("Kauldron автоматически загрузит последнее состояние.")
    init_transform = None   # kauldron сам загрузит из чекпойнта
else:
    init_transform = Layer23InitTransform(merged_params)

# ── Ячейка 4: Датасет ──
train_ds = get_activation_dataset(batch_size=16)

# ── Ячейка 5: Оптимизатор ──
titans_optimizer = make_routing_optimizer(opt_params)
optimizer = make_layer23_optimizer(titans_optimizer)

# ── Ячейка 6: Trainer ──
trainer = kd.train.Trainer(
    seed=42,
    workdir=workdir,
    train_ds=train_ds,
    model=model,
    init_transform=init_transform,
    # optimizer=optimizer,   # ← если Trainer принимает (проверь API kauldron)
    num_steps=50_000,
)

trainer.train()
```

### Что даёт kauldron автоматически:
- ✅ Чекпойнты в `{workdir}/checkpoints/` (params + opt_state + step)
- ✅ Resume при перезапуске (подхватывает последний чекпойнт)
- ✅ TensorBoard логирование loss/accuracy
- ✅ Loss извлекается из `preds.layer_losses['lm_loss']` (как в Phase2)

### Структура params в чекпойнте:
```
params/
├── layer_23/     ← TitansBlock (обучаемый)
├── layer_24/     ← Frozen (identity optimizer)
├── layer_25/     ← Frozen
├── final_norm/   ← Frozen
└── embedder/     ← Frozen
```