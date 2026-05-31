"""
Kauldron-compatible model for training TitansBlock (layer 23+)
with precomputed activations from parquet dataset.

Layer23Model subclasses Gemma3_1B_Titans, reusing its full
_apply_attention / _lm_loss / TitansBlock logic.
Only setup() and __call__() are overridden:
  - setup(): calls super(), then slices self.blocks[titans_first_layer:]
  - __call__(): accepts activations + tokens + mask instead of tokens only
"""

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import grain.python as grain
from flax.core import freeze, unfreeze

from kauldron import kd
from kauldron import kontext

from gemma.gm.nn import _transformer
from gemma.gm.utils import _jax_utils

from gemma_titans import Gemma3_1B_Titans, Gemma_Titans_Config, TrainingOutput
from titans import init_memory_state


# ──────────────────────────── Model ─────────────────────────────

class Layer23Model(Gemma3_1B_Titans):
    """
    Subclass of Gemma3_1B_Titans that accepts precomputed activations.

    setup() calls super().setup() then slices self.blocks to keep only
    blocks from titans_first_layer onward.  All parent logic
    (_apply_attention, Phase 2 loss, etc.) is reused unchanged.

    To train more TitansBlock layers, change titans_first_layer in config:
        23 → layers 23,24,25 (1 TitansBlock + 2 Gemma)
        17 → layers 17,18,...,25 (3 TitansBlock + 6 Gemma)
        11 → layers 11,12,...,25 (5 TitansBlock + 10 Gemma)
    """

    # Kauldron kontext keys — activations come from parquet dataset
    hidden: kontext.Key = "batch.activations"
    tokens: kontext.Key = "batch.tokens"
    mask:   kontext.Key = "batch.mask"
    step:   kontext.Key = "step"

    # ── setup: reuse parent, then slice blocks ────────────────
    def setup(self):
        super().setup()
        first = self.config.titans_first_layer
        self.blocks = self.blocks[first:]
        print(f"Layer23Model: sliced blocks to [{first}..{first+len(self.blocks)-1}] "
              f"({len(self.blocks)} blocks)")

    # ── forward: accept precomputed activations ───────────────
    def __call__(
        self,
        hidden,
        tokens,
        mask,
        *,
        step,
        **kwargs,
    ) -> Union[TrainingOutput, _transformer.Output]:
        # Cast activations to model's dtype (e.g. bfloat16)
        hidden = hidden.astype(self.dtype)
        B, L, D = hidden.shape

        # step handling (same as parent)
        step_b = jnp.broadcast_to(
            jnp.asarray(step, dtype=jnp.int32),
            (B,),  # (B,)
        )

        # Construct _Inputs from precomputed activations + mask
        # (replaces parent's _encode_and_get_inputs which does token embedding)
        positions = jnp.broadcast_to(jnp.arange(L)[None, :], (B, L))
        causal = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
        attention_mask = causal[None, :, :] & mask[:, None, :].astype(jnp.bool_)
        inputs_mask = mask.astype(jnp.float32)

        inputs = SimpleNamespace(
            embeddings=hidden,
            positions=positions,
            attention_mask=attention_mask,
            inputs_mask=inputs_mask,
        )

        # step scalar for huber delta schedule
        step_scalar = step_b[0]

        # Evaluate huber delta (same as parent)
        current_huber_delta = None
        huber_delta_cfg = self.config.neural_mem_kwargs.get('huber_loss_delta')
        if huber_delta_cfg is not None:
            current_huber_delta = huber_delta_cfg(step_scalar) if callable(huber_delta_cfg) else huber_delta_cfg

        # Reuse parent's _apply_attention (TitansBlock + Gemma blocks + final_norm)
        x, new_cache, layer_losses = self._apply_attention(
            inputs,
            None,  # cache — not used during training
            is_training=self.config.is_training_mode,
            current_huber_delta=current_huber_delta,
        )

        # Phase 2: LM loss — compute inline (no @jax.checkpoint to avoid
        # UnexpectedTracerError: self.embedder captures tracers that leak
        # from jax.checkpoint scope when __call__ lacks @flax_nn.jit)
        if self.config.is_training_mode and self.config.training_phase == 2:
            logits = self.embedder.decode(x[:, :-1, :])
            if self.config.final_logit_softcap is not None:
                logits /= self.config.final_logit_softcap
                logits = jnp.tanh(logits) * self.config.final_logit_softcap

            tgt = tokens[:, 1:]
            valid_mask = inputs_mask[:, 1:].astype(jnp.float32)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), tgt
            )
            denom = jnp.maximum(valid_mask.sum(axis=-1), 1.0)
            lm_loss = (ce * valid_mask).sum(axis=-1) / denom
            pred = jnp.argmax(logits, axis=-1)
            lm_acc = (
                ((pred == tgt).astype(jnp.float32) * valid_mask).sum(axis=-1)
                / denom
            )
            layer_losses['lm_loss'] = lm_loss
            layer_losses['lm_accuracy'] = lm_acc

            return TrainingOutput(
                logits=jnp.zeros((B, 1)),
                cache=None,
                hidden_states=None,
                layer_losses=layer_losses,
            )
        else:
            # Non-training or Phase 1 — shouldn't normally happen for this model
            return TrainingOutput(
                logits=jnp.zeros((B, 1)),
                cache=None,
                hidden_states=x,
                layer_losses=layer_losses,
            )


# ──────────────────── Init Transform ────────────────────────────

class Layer23InitTransform(kd.ckpts.InitTransform):
    """Loads merged_params, keeps only layers from titans_first_layer onward + embedder.

    Args:
        merged_params: Full model params (Gemma + Titans weights merged).
        titans_first_layer: First layer index to keep (default 23).
        random_init_titans: If True, TitansBlock layers (layer_23 etc.)
            are initialized from the model's random init state instead of
            merged_params. Frozen layers still come from merged_params.
            Useful for ablation / from-scratch training.
    """

    def __init__(self, merged_params, titans_first_layer=23, random_init_titans=False, dtype=jnp.bfloat16):
        self.merged_params = unfreeze(merged_params)
        self.titans_first_layer = titans_first_layer
        self.random_init_titans = random_init_titans
        self.dtype = dtype

        # Standard Titans layer indices
        self._all_titans = (11, 17, 23)
        self._active_titans = {f'layer_{l}' for l in self._all_titans if l >= titans_first_layer}

    def _cast_to_dtype(self, params):
        """Cast all parameter arrays to self.dtype."""
        return jax.tree.map(lambda x: x.astype(self.dtype) if hasattr(x, 'astype') else x, params)

    def transform(self, state):
        p = self.merged_params
        num_layers = 26  # Gemma 1B has 26 layers (0..25)

        # Start with frozen layers from merged_params
        params = {}
        for i in range(self.titans_first_layer, num_layers):
            key = f'layer_{i}'
            if key in p:
                if self.random_init_titans and key in self._active_titans:
                    # Keep Flax's built-in init (lecun_normal for kernels, constant for biases)
                    if key in state.params:
                        params[key] = self._cast_to_dtype(unfreeze(state.params[key]))
                        print(f"  🎲 {key}: random init (TitansBlock, Flax defaults) → {self.dtype}")
                    continue
                params[key] = self._cast_to_dtype(p[key])
        params['final_norm'] = self._cast_to_dtype(p['final_norm'])
        params['embedder'] = self._cast_to_dtype(p['embedder'])
        return state.replace(params=freeze(params))


# ──────────────────── Masked Optimizer ──────────────────────────

def make_layer23_optimizer(titans_optimizer, titans_first_layer=23):
    """Only TitansBlock layers are trainable, all other params frozen.

    TitansBlock layers are those in config.titans_layer_indices that
    are >= titans_first_layer. All other layers get identity optimizer.
    """
    # Standard Titans layer indices
    all_titans = (11, 17, 23)
    active_titans = tuple(l for l in all_titans if l >= titans_first_layer)

    active_titans_str = {f'layer_{l}' for l in active_titans}

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: 'train' if path[0].key in active_titans_str else 'freeze',
            params)
    return optax.multi_transform(
        {'train': titans_optimizer, 'freeze': optax.identity()},
        label_fn)


# ──────────────────── Dataset Pipeline ──────────────────────────

class _ProcessActivations(grain.MapTransform):
    """Convert raw parquet fields to properly shaped numpy arrays."""
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
    cache_dir= None
):
    """Kauldron dataset pipeline for precomputed activations."""
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
        ],
        cache_dir= cache_dir
        )