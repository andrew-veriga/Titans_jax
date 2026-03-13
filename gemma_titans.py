import dataclasses
import functools
import typing
from typing import Any, ClassVar, Optional, Tuple, Dict, Union

import jax
import jax.numpy as jnp
import flax
from flax import struct
from flax import linen as nn

# Official Gemma framework imports
from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
from gemma.gm.nn import _transformer
from gemma.gm.nn import _gemma
from gemma.gm.utils import _jax_utils
from gemma.gm.utils import _dtype_params
from gemma.gm.vision import _token_utils
from kauldron.typing import Bool, Float, Int, UInt8
import os
os.environ['KAULDRON_TYPECHECK'] = '0'
os.environ['KD_CHECK_TYPES'] = '0'

# Import the existing Neural Memory from the project
from titans import NeuralMemory, init_memory_state

@struct.dataclass
class TitansLayerCache:
    attention_cache: Optional[_modules.LayerCache]
    memory_state: Optional[Any] # State for NeuralMemory

class TitansBlock(_modules.Block):
    """Gemma Block with integrated Titans Neural Long-Term Memory (NLTM)."""
    
    def setup(self):
        super().setup()
        
        self.memory = NeuralMemory(
            dim=self.embed_dim,
            heads=self.num_heads,
            dim_head=64,
            chunk_size=16,
        )
        
        self.memory_gate = self.param('memory_gate', nn.initializers.zeros, (1,))

    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: Optional[Union[_modules.LayerCache, TitansLayerCache]],
        attn_mask: jax.Array,
    ) -> tuple[Optional[TitansLayerCache], jax.Array]:
        
        # Handle different cache types
        if isinstance(cache, dict) and not isinstance(cache, TitansLayerCache):
            attn_cache = cache
            mem_state = None
        elif isinstance(cache, TitansLayerCache):
            attn_cache = cache.attention_cache
            mem_state = cache.memory_state
        else:
            attn_cache = None
            mem_state = None

        inputs_normalized = self.pre_attention_norm(x)

        # 1. Standard Attention Branch
        new_attn_cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            attn_cache,
            attn_mask,
        )

        # 2. Neural Memory (Titans) Branch
        retrieved, next_mem_state = self.memory(
            inputs_normalized,
            memory_state=mem_state,
            return_next_memories=True
        )

        # Combine Attention and Memory
        combined_output = attn_output + jax.nn.sigmoid(self.memory_gate) * retrieved

        if self.post_attention_norm is not None:
            combined_output = self.post_attention_norm(combined_output)

        combined_output += x

        # 3. MLP Branch
        outputs = self.pre_ffw_norm(combined_output)
        outputs = self.mlp(outputs)

        if self.post_ffw_norm is not None:
            outputs = self.post_ffw_norm(outputs)

        outputs += combined_output

        # Construct new cache
        new_cache = TitansLayerCache(
            attention_cache=new_attn_cache,
            memory_state=next_mem_state
        )

        return new_cache, outputs


class Gemma3_1B_Titans(_gemma.Gemma3_1B):
    """Gemma3 1B with integrated Titans NLTM."""
    
    def setup(self):
        self.embedder = _modules.Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
            vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
            if self.config.vision_encoder
            else None,
        )

        self.blocks = [
            nn.remat(TitansBlock)(
                name=f'layer_{i}',
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                embed_dim=self.config.embed_dim,
                head_dim=self.config.head_dim,
                hidden_dim=self.config.hidden_dim,
                sliding_window_size=self.config.sliding_window_size,
                use_post_attn_norm=self.config.use_post_attn_norm,
                use_post_ffw_norm=self.config.use_post_ffw_norm,
                attn_logits_soft_cap=self.config.attn_logits_soft_cap,
                attn_type=attn_type,
                query_pre_attn_scalar=self.config.query_pre_attn_scalar(),
                transpose_gating_einsum=self.config.transpose_gating_einsum,
                use_qk_norm=self.config.use_qk_norm,
                rope_base_frequency=self.config.local_base_frequency
                if attn_type == _modules.AttentionType.LOCAL_SLIDING
                else self.config.global_base_frequency,
                rope_scale_factor=self.config.local_scale_factor
                if attn_type == _modules.AttentionType.LOCAL_SLIDING
                else self.config.global_scale_factor,
            )
            for i, attn_type in zip(
                range(self.config.num_layers), self.config.attention_types
            )
        ]
        self.final_norm = _layers.RMSNorm()

        self.vision_encoder = self.config.vision_encoder

    @functools.partial(
        nn.jit,
        static_argnames=(
            'self',
            'return_last_only',
            'return_hidden_states',
        ),
    )
    @_jax_utils.flatten_unflatten_batch_dim()
    def __call__(
        self,
        tokens: Int['*B L'],
        *,
        images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
        positions: Int['*B L_with_mm'] | None = None,
        cache: Optional[Dict[str, TitansLayerCache]] = None,
        attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
        return_last_only: bool | None = None,
        return_hidden_states: bool | None = None,
    ) -> _transformer.Output:
        """Forward pass - copied from base class to bypass @typechecked on cache."""
        return_last_only = self._get_return_last_only(return_last_only)

        with _dtype_params.initialize_param_with_dtype(
            self.dtype,
            exclude=[
                'vision_encoder',
                'embedder.mm_input_projection',
                'embedder.mm_soft_embedding_norm',
                'lora',
            ],
        ):
            inputs = self._encode_and_get_inputs(
                tokens=tokens,
                images=images,
                positions=positions,
                attention_mask=attention_mask,
            )
            del positions, attention_mask

            x, new_cache = self._apply_attention(inputs, cache)

        if return_last_only:
            last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
            x = x[jnp.arange(len(x)), last_input_token_idx, ...]
        elif images is not None:
            x = _token_utils.remove_mm_logits(
                logits=x,
                tokens=tokens,
                num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,
            )

        logits = self.embedder.decode(x)

        if self.config.final_logit_softcap is not None:
            logits /= self.config.final_logit_softcap
            logits = jnp.tanh(logits) * self.config.final_logit_softcap

        return _transformer.Output(
            logits=logits,
            cache=None if cache is None else new_cache,
            hidden_states=x if return_hidden_states else None,
        )

    def _apply_attention(
        self, inputs: _transformer._Inputs, cache: Optional[Dict[str, TitansLayerCache]]
    ) -> tuple[jax.Array, Dict[str, TitansLayerCache]]:
        x = inputs.embeddings
        old_cache = cache or {}
        new_cache = {}
        for i, block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            layer_cache, x = block(
                x,
                inputs.positions,
                old_cache.get(layer_name),
                inputs.attention_mask,
            )
            new_cache[layer_name] = layer_cache
            
        x = self.final_norm(x)
        return x, new_cache

    def init_cache(
        self,
        *,
        batch_size: int,
        dtype: jnp.dtype,
        cache_length: int,
        sharding: Any = None,
    ) -> Dict[str, TitansLayerCache]:
        
        cache = {}
        for i in range(self.config.num_layers):
            layer_name = f'layer_{i}'
            
            attn_cache = _modules.Attention.init_cache(
                cache_size=cache_length,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                batch_size=batch_size,
                dtype=dtype
            )
            
            mem_state = init_memory_state(
                batch_size=batch_size,
                dim=self.config.embed_dim,
                heads=self.config.num_heads
            )
            
            cache[layer_name] = TitansLayerCache(
                attention_cache=attn_cache,
                memory_state=mem_state
            )
            
        return cache
