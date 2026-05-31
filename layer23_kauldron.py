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
from gemma.gm.utils import _dtype_params

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
        self._block_offset = first  # remember offset for _apply_attention
        print(f"Layer23Model: sliced blocks to [{first}..{first+len(self.blocks)-1}] "
              f"({len(self.blocks)} blocks)")

    # ── _apply_attention: fix layer indices after block slicing ──
    def _apply_attention(self, inputs, cache, is_training, current_huber_delta=None):
        """Override parent: adjust layer indices for sliced blocks."""
        x = inputs.embeddings
        x_prev = x

        old_cache = cache or {}
        new_cache = {}
        layer_losses = {}

        # Build student mask (truncated window) — same as parent
        if inputs.attention_mask is not None:
            k_len = inputs.attention_mask.shape[-1]
            window = 128
            q_pos = inputs.positions[:, :, None]
            k_pos = jnp.arange(k_len, dtype=jnp.int32)[None, None, :]
            sliding_window = (q_pos - k_pos) < window
            s_mask = inputs.attention_mask & sliding_window
        else:
            s_mask = None

        from gemma_titans import TitansBlock

        offset = self._block_offset
        for i, block in enumerate(self.blocks):
            real_i = i + offset  # actual layer index in full model
            layer_name = f'layer_{real_i}'

            if isinstance(block, TitansBlock):
                if is_training and self.config.training_phase == 1:
                    # Phase 1 distillation (not typical for Layer23 but kept for completeness)
                    layer_cache_teacher, out_teacher = block(
                        x, inputs.positions, old_cache.get(layer_name),
                        inputs.attention_mask, is_teacher_mode=True,
                        kv_seq=x_prev if block.diff_view else None,
                        current_huber_delta=current_huber_delta,
                    )
                    layer_cache_student, out_student = block(
                        jax.lax.stop_gradient(x), inputs.positions,
                        old_cache.get(layer_name), s_mask,
                        is_teacher_mode=False,
                        kv_seq=jax.lax.stop_gradient(x_prev) if block.diff_view else None,
                        current_huber_delta=current_huber_delta,
                    )
                    if layer_cache_student is not None and 'avg_mem_loss' in layer_cache_student:
                        layer_losses[f"mem_loss_{layer_name}"] = layer_cache_student['avg_mem_loss']
                    else:
                        layer_losses[f"mem_loss_{layer_name}"] = jnp.zeros((x.shape[0],), dtype=jnp.float32)
                    delta_teacher = jax.lax.stop_gradient(out_teacher - x)
                    delta_student = out_student - x
                    layer_loss = self.cos_by_softmax(delta_teacher, delta_student)
                    layer_loss = layer_loss * inputs.inputs_mask.astype(layer_loss.dtype)
                    layer_losses[f"loss_{layer_name}"] = layer_loss
                    if layer_cache_student is not None and 'gate_values' in layer_cache_student:
                        layer_losses[f"gate_{layer_name}"] = layer_cache_student['gate_values']
                    else:
                        layer_losses[f"gate_{layer_name}"] = jnp.zeros_like(x)
                    x_prev = x
                    x = out_teacher
                    if layer_cache_teacher is not None:
                        merged_cache = dict(layer_cache_teacher)
                        if layer_cache_student is not None and 'memory_state' in layer_cache_student:
                            merged_cache['memory_state'] = layer_cache_student['memory_state']
                        new_cache[layer_name] = merged_cache
                    else:
                        new_cache[layer_name] = None
                else:
                    # Phase 2 / Inference: stop_gradient at first TitansBlock
                    if real_i == self.config.titans_first_layer and is_training:
                        x = jax.lax.stop_gradient(x)
                        x_prev = jax.lax.stop_gradient(x_prev)

                    layer_cache_student, out_student = block(
                        x, inputs.positions, old_cache.get(layer_name),
                        s_mask if s_mask is not None else inputs.attention_mask,
                        False,  # is_teacher_mode
                        x_prev if block.diff_view else None,
                        current_huber_delta=current_huber_delta,
                    )
                    if layer_cache_student is not None and 'avg_mem_loss' in layer_cache_student:
                        layer_losses[f"mem_loss_{layer_name}"] = layer_cache_student['avg_mem_loss']
                    x_prev = x
                    x = out_student
                    new_cache[layer_name] = layer_cache_student
            else:
                # Standard Gemma Block
                layer_cache, out_next = block(
                    x, inputs.positions, old_cache.get(layer_name),
                    inputs.attention_mask,
                )
                x_prev = x
                x = out_next
                new_cache[layer_name] = layer_cache

        x = self.final_norm(x)

        # Phase 1 total distillation loss (same as parent)
        if self.config.training_phase == 1 and is_training:
            mask_float = inputs.inputs_mask.astype(jnp.float32)
            mask_count = jnp.maximum(mask_float.sum(axis=-1), 1.0)
            total_distill = jnp.zeros((x.shape[0],), dtype=jnp.float32)
            count = 0
            for k, v in layer_losses.items():
                if k.startswith("loss_layer_"):
                    total_distill = total_distill + v.astype(jnp.float32).sum(axis=-1) / mask_count
                    count += 1
            if count > 0:
                layer_losses['lm_loss'] = total_distill / count
            else:
                layer_losses['lm_loss'] = jnp.zeros((x.shape[0],), dtype=jnp.float32)

        return x, new_cache, layer_losses

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

        # Use same dtype context as parent _forward — ensures all Flax
        # parameter lookups use the correct dtype (e.g. bfloat16).
        # Without this, Flax may use float32 internally, causing dtype
        # mismatches and potential NaN from mixed-precision issues.
        with _dtype_params.initialize_param_with_dtype(
            self.dtype,
            exclude=[
                'vision_encoder',
                'embedder.mm_input_projection',
                'embedder.mm_soft_embedding_norm',
                'lora',
            ],
        ):
            # Reuse parent's _apply_attention (TitansBlock + Gemma blocks + final_norm)
            x, new_cache, layer_losses = self._apply_attention(
                inputs,
                None,  # cache — not used during training
                is_training=self.config.is_training_mode,
                current_huber_delta=current_huber_delta,
            )

            # Phase 2: LM loss — inside dtype context so embedder uses correct dtype
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