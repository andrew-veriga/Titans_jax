import dataclasses
import functools
import typing
from typing import Any, ClassVar, Optional, Tuple, Dict, Union

import jax
import jax.numpy as jnp
import flax
from flax import struct
from flax import linen as flax_nn

# Official Gemma framework imports
from gemma.gm.nn import _config
from gemma.gm.nn import _layers
from gemma.gm.nn import _modules
from gemma.gm.nn import _transformer
from gemma.gm.nn import _gemma
from gemma.gm.utils import _jax_utils
from gemma.gm.utils import _dtype_params
from gemma.gm.vision import _token_utils
from gemma.gm.utils import _cache_helper
from kauldron.typing import Bool, Float, Int, UInt8
import os

os.environ['KAULDRON_TYPECHECK'] = '0'
os.environ['KD_CHECK_TYPES'] = '0'
import importlib
import titans
importlib.reload(titans)

# Import the existing Neural Memory from the project
from titans import NeuralMemory, init_memory_state

# Monkeypatch _set_cache to support memory_state merging during prefill
# To safely handle Colab cell re-runs without recursion, check if the function is already patched.
if not hasattr(_cache_helper._set_cache, '_is_titans_patched'):
    _orig_set_cache = _cache_helper._set_cache

    def _new_set_cache(layer_data0, layer_data1, *, key):
        layer_data0 = _orig_set_cache(layer_data0, layer_data1, key=key)
        if 'memory_state' in layer_data1:
            layer_data0['memory_state'] = layer_data1['memory_state']
        return layer_data0
    
    _new_set_cache._is_titans_patched = True
    _cache_helper._set_cache = _new_set_cache

class TitansBlock(_modules.Block):
    """Gemma Block with integrated Titans Neural Long-Term Memory (NLTM)."""
    
    def setup(self):
        super().setup()
        
        self.memory = NeuralMemory(
            dim=self.embed_dim,
            heads=self.num_heads,
            dim_head=128,
            chunk_size=16,
        )
        
        # 1152 независимых вентиля
        self.memory_gate = self.param(
            'memory_gate', 
            flax_nn.initializers.constant(-2.0), 
            (self.embed_dim,) # <-- Ключевое изменение: задаем размерность эмбеддинга
        )
        
    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: Optional[Dict[str, Any]],
        attn_mask: jax.Array,
        is_teacher_mode: bool = False,
        **kwargs,
    ) -> tuple[Optional[Dict[str, Any]], jax.Array]:
        
        if cache is not None:
            mem_state = cache.get('memory_state')
        else:
            mem_state = None

        inputs_normalized = self.pre_attention_norm(x)

        # 1. Standard Attention Branch
        new_attn_cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )

        if is_teacher_mode:
            # Teacher mode: Full attention, no memory, residual
            combined_output = attn_output
            next_mem_state = mem_state # Memory state doesn't change in teacher mode
        else:
            # 2. Neural Memory (Titans) Branch (Student mode)
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
        if cache is not None:
            new_cache = dict(new_attn_cache)
            new_cache['memory_state'] = next_mem_state
        else:
            new_cache = None

        return new_cache, outputs

@dataclasses.dataclass(frozen=True)
class Gemma_Titans_Config(_config.TransformerConfig):
    """Configuration for Gemma3 with Titans NLTM."""
    titans_layer_indices: tuple[int, ...] = (5, 11, 17, 23)
    is_training_mode: bool = True

@flax.struct.dataclass
class DistillationOutput:
    logits: jax.Array
    cache: Optional[Dict[str, Any]]
    hidden_states: Optional[jax.Array]
    layer_losses: Dict[str, jax.Array] = struct.field(default_factory=dict)


class Gemma3_1B_Titans(_gemma.Gemma3_1B):
    """Gemma3 1B with integrated Titans NLTM (Bimodal: Training & Inference)."""
    
    config: Gemma_Titans_Config = Gemma_Titans_Config(
        **{f.name: getattr(_gemma.Gemma3_1B.config, f.name) 
           for f in dataclasses.fields(_config.TransformerConfig)}
    )

    def setup(self):
        self.embedder = _modules.Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
            vision_proj_dim=self.config.vision_encoder.siglip_encoder.width
            if self.config.vision_encoder
            else None,
        )

        blocks = []
        num_layers = len(self.config.attention_types)
        for i, attn_type in zip(range(num_layers), self.config.attention_types):
            block_kwargs = dict(
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
            
            if i in self.config.titans_layer_indices:
                blocks.append(TitansBlock(**block_kwargs))
            else:
                blocks.append(_modules.Block(**block_kwargs))
                
        self.blocks = blocks
        self.final_norm = _layers.RMSNorm()

        self.vision_encoder = self.config.vision_encoder

    @functools.partial(
        flax_nn.jit,
        static_argnames=(
            'self',
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
        cache: Optional[Dict[str, Any]] = None,
        attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
        # loss_mask: jax.Array | None = None, # <--- ПРИНИМАЕМ МАСКУ ИЗ БАТЧА ДЛЯ ОБУЧЕНИЯ
        return_hidden_states: bool | None = None,
        **kwargs,
    ) -> Union[DistillationOutput, _transformer.Output]:
        """Forward pass - automatically switches between Meta-Training and Inference."""
        return_last_only = self.return_last_only
        
        # MАРКЕР РЕЖИМА: Если есть loss_mask, значит мы в цикле тренировки Kauldron
        is_training = self.config.is_training_mode

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

            x, new_cache, layer_losses = self._apply_attention(
                inputs, 
                cache, 
                is_training=is_training
            )

        if is_training:
            # РЕЖИМ ТРЕНИРОВКИ (Дистилляция)
            # Отключаем декодер логитов для экономии памяти (сохраняем ~16GB HBM)
            return DistillationOutput(
                logits=jnp.zeros((x.shape[0], 1)), # Заглушка
                cache=None if cache is None else new_cache,
                hidden_states=x if return_hidden_states else None,
                layer_losses=layer_losses
            )
        else:
            # РЕЖИМ ИНФЕРЕНСА (Генерация текста)
            if return_last_only:
                last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
                x = x[jnp.arange(len(x)), last_input_token_idx, ...]
            elif images is not None:
                x = _token_utils.remove_mm_logits(
                    logits=x,
                    tokens=tokens,
                    num_tokens_per_image=self.config.vision_encoder.num_mm_tokens_per_image,
                )

            # Честно вычисляем предсказания словаря
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
        self, 
        inputs: _transformer._Inputs, 
        cache: Optional[Dict[str, Any]],
        # loss_mask: jax.Array | None,
        is_training: bool,
    ) -> tuple[jax.Array, Dict[str, Any], Dict[str, jax.Array]]:
        x = inputs.embeddings
        old_cache = cache or {}
        new_cache = {}
        layer_losses = {}
        
        # Build student mask (truncated window)
        if inputs.attention_mask is not None:
            seq_len = x.shape[-2]
            window = 128  # Truncated context for Student
            
            row_idx = jnp.arange(seq_len)[:, None]
            col_idx = jnp.arange(seq_len)[None, :]
            sliding_window = (row_idx - col_idx) < window
            
            s_mask = inputs.attention_mask & sliding_window
        else:
            s_mask = None

        for i, block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            
            if isinstance(block, TitansBlock):
                if is_training:
                    # 1. Teacher Pass (Full context)
                    layer_cache_teacher, out_teacher = block(
                        x,
                        inputs.positions,
                        old_cache.get(layer_name),
                        inputs.attention_mask, # teacher mask
                        is_teacher_mode=True
                    )
                    
                    # 2. Student Pass (Truncated context, using the SAME input 'x')
                    layer_cache_student, out_student = block(
                        x,
                        inputs.positions,
                        old_cache.get(layer_name),
                        s_mask, # student mask
                        is_teacher_mode=False
                    )
                    
                    # 3. Layer Loss
                    raw_diff = (out_student - jax.lax.stop_gradient(out_teacher)) ** 2
                    
                    # if loss_mask is not None:
                    #   safe_mask = loss_mask.reshape(x.shape[0], x.shape[1], 1)
                    #   raw_diff = raw_diff * safe_mask
                    
                    layer_loss = jnp.mean(raw_diff, axis=(1, 2), dtype=jnp.float32)
                    
                    layer_losses[f"loss_{layer_name}"] = jnp.log1p(layer_loss)
                    layer_losses[f"raw_mse_{layer_name}"] = layer_loss
                    
                    # 4. Update x with Teacher's output to prevent Exposure Bias
                    x = out_teacher
                    new_cache[layer_name] = layer_cache_teacher
                else:
                    # INFERENCE MODE: Only run Student Pass
                    layer_cache_student, out_student = block(
                        x,
                        inputs.positions,
                        old_cache.get(layer_name),
                        s_mask, # student mask
                        is_teacher_mode=False
                    )
                    x = out_student
                    new_cache[layer_name] = layer_cache_student
            else:
                # Standard Gemma Block
                layer_cache, x = block(
                    x,
                    inputs.positions,
                    old_cache.get(layer_name),
                    inputs.attention_mask,
                )
                new_cache[layer_name] = layer_cache
            
        x = self.final_norm(x)
        return x, new_cache, layer_losses

    def init_cache(
        self,
        *,
        batch_size: int,
        dtype: jnp.dtype,
        cache_length: int,
        sharding: Any = None,
    ) -> Dict[str, Any]:
        
        cache = {}
        num_layers = len(self.config.attention_types)
        for i in range(num_layers):
            layer_name = f'layer_{i}'
            
            attn_cache = _modules.Attention.init_cache(
                cache_size=cache_length,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                batch_size=batch_size,
                dtype=dtype
            )
            
            if i in self.config.titans_layer_indices:
                mem_state = init_memory_state(
                    batch_size=batch_size,
                    dim=self.config.embed_dim,
                    heads=self.config.num_heads,
                    dim_head=128,
                    dtype=dtype
                )
                attn_cache['memory_state'] = mem_state
            cache[layer_name] = attn_cache
            
        return cache
