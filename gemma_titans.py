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
from kauldron import kontext
import os

os.environ['KAULDRON_TYPECHECK'] = '0'
os.environ['KD_CHECK_TYPES'] = '0'
import importlib
import optax
from optax._src import base
import titans
importlib.reload(titans)

# Import the existing Neural Memory from the project
from titans import NeuralMemory, init_memory_state, huber_loss, default_loss_fn

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
    diff_view: bool = False # If True, the QKV projections in the TitansBlock receive the "diff view" input (previous layer's output) instead of the current layer's input. This can help stabilize training by providing a more consistent signal to the memory across layers.
    elastic_net_lambda: Optional[float] = None
    huber_loss_delta: base.ScalarOrSchedule = None
    neural_mem_heads: int = 8
    is_look_ahead: bool = False
    use_original_attn: bool = False

    def setup(self):
        from gemma.gm.nn import _layers
        self.pre_attention_norm = _layers.RMSNorm()

        self.attn = None
        if self.use_original_attn:
            self.attn = _modules.Attention(
                num_heads=self.num_heads,
                features=self.embed_dim,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                attn_type=self.attn_type,
                query_pre_attn_scalar=self.query_pre_attn_scalar,
                rope_base_frequency=self.rope_base_frequency,
                rope_scale_factor=self.rope_scale_factor,
                attn_logits_soft_cap=self.attn_logits_soft_cap,
                sliding_window_size=self.sliding_window_size,
                use_qk_norm=self.use_qk_norm,
            )

        self.post_attention_norm = None
        if self.use_post_attn_norm:
            self.post_attention_norm = _layers.RMSNorm()

        self.pre_ffw_norm = _layers.RMSNorm()

        self.mlp = _modules.FeedForward(
            features=self.embed_dim,
            hidden_dim=self.hidden_dim,
            transpose_gating_einsum=self.transpose_gating_einsum,
        )

        self.post_ffw_norm = None
        if self.use_post_ffw_norm:
            self.post_ffw_norm = _layers.RMSNorm()

        # Note: huber_loss_delta is now evaluated per-call if it's a schedule
        self.memory = NeuralMemory(
            dim=self.embed_dim,
            heads=self.neural_mem_heads,
            dim_head=256,
            chunk_size=32,
            max_grad_norm=0.5,
            elastic_net_lambda=self.elastic_net_lambda,
            diff_view=self.diff_view,
            is_look_ahead=self.is_look_ahead,
            store_memory_loss_fn=huber_loss if self.huber_loss_delta is not None else default_loss_fn
        )
        
        # 1152 независимых вентиля
        # ДИНАМИЧЕСКИЙ ВЕНТИЛЬ: вместо статического параметра используем Dense-слой
        # для вычисления важности памяти на основе текущего токена
        self.memory_gate_proj = flax_nn.Dense(
            features=self.embed_dim, 
            use_bias=False,
            name='memory_gate_proj'
        )
        
    def __call__(
        self,
        x: jax.Array,
        segment_pos: jax.Array,
        cache: Optional[Dict[str, Any]],
        attn_mask: jax.Array,
        is_teacher_mode: bool = False,
        kv_seq: Optional[jax.Array] = None,
        current_huber_delta: Optional[Union[float, jax.Array]] = None,
        **kwargs,
    ) -> tuple[Optional[Dict[str, Any]], jax.Array]:
        
        if cache is not None:
            mem_state = cache.get('memory_state')
        else:
            mem_state = None

        inputs_normalized = self.pre_attention_norm(x)

        if is_teacher_mode:
            # 1. Teacher mode (Phase 1 target): Full attention Gemma, no memory
            new_attn_cache, attn_output = self.attn(
                inputs_normalized,
                segment_pos,
                cache,
                attn_mask,
            )
            combined_output = attn_output
            next_mem_state = mem_state # Memory state doesn't change in teacher mode
        else:
            # 2. Student mode (Phase 1) / Phase 2 / Inference:
            # Pure Titans memory, NO original Gemma attention
            
            loss_kwargs = {}
            if current_huber_delta is not None:
                loss_kwargs['delta'] = current_huber_delta

            retrieved, next_mem_state = self.memory(
                inputs_normalized,
                memory_state=mem_state,
                return_next_memories=True,
                kv_seq=kv_seq,
                loss_kwargs=loss_kwargs
            )
            # Динамический вентиль: вычисляется на основе входного вектора
            gate = jax.nn.sigmoid(jnp.clip(self.memory_gate_proj(inputs_normalized), -10.0, 10.0))
            
            # ЧИСТАЯ ЗАМЕНА: Оригинальный attn_output больше не прибавляется!
            combined_output = gate * retrieved
            
            # Пробрасываем старый кэш Attention без изменений, чтобы сохранить структуру PyTree
            new_attn_cache = dict(cache) if cache is not None else {}
            # Удаляем memory_state из копии attn_cache, так как мы добавим его ниже
            if 'memory_state' in new_attn_cache:
                del new_attn_cache['memory_state']

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
    neural_mem_qkv_receives_diff_view: bool = True # If True, the QKV projections in the TitansBlock receive the "diff view" input (previous layer's output) instead of the current layer's input. This can help stabilize training by providing a more consistent signal to the memory across layers.
    training_phase: int = 2  # 1: per-layer distillation, 2: LM fine-tuning and inference
    # Phase 2 only: stop_gradient is inserted before this layer to limit backward graph depth.
    # Must be one of titans_layer_indices. Smaller value = deeper backward = more compile RAM.
    # 23 → backward through 3 layers (~5GB compile RAM)
    # 17 → backward through 9 layers (~25GB compile RAM)
    # 11 → backward through 15 layers (~70GB compile RAM)
    titans_phase2_first_layer: int = 23
    neural_mem_elastic_lambda: Optional[float] = None
    neural_mem_huber_delta: base.ScalarOrSchedule = None
    neural_mem_heads: int = 8  # Must match TitansBlock NeuralMemory(heads=...)
    is_look_ahead: bool = False
    @classmethod
    def from_gemma_config(
        cls, 
        ):
        """
        Правильный способ 'заливки': берем поля из gemma_config,
        но фильтруем только те, что есть в текущем классе.
        """
        # Список имен полей, которые принимает наш конструктор
        valid_field_names = {f.name for f in dataclasses.fields(cls)}
        
        # Извлекаем значения из оригинального конфига
        config_dict = { 
            f.name: getattr(_gemma.Gemma3_1B.config, f.name) 
            for f in dataclasses.fields(_config.TransformerConfig) 
            if f.name in valid_field_names 
        }
        return cls(**config_dict)

        
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
    
    tokens: kontext.Key = "batch.tokens"
    step: kontext.Key = "step"

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
            
            if i in self.config.titans_layer_indices and i >= self.config.titans_phase2_first_layer:
                if self.config.training_phase == 1:
                        blocks.append(TitansBlock(
                        **block_kwargs,
                        diff_view=self.config.neural_mem_qkv_receives_diff_view,
                        elastic_net_lambda=self.config.neural_mem_elastic_lambda,
                        huber_loss_delta=self.config.neural_mem_huber_delta,
                        neural_mem_heads=self.config.neural_mem_heads,
                        use_original_attn=True, # Phase 1 requires Gemma Attention for Teacher
                    ))
                else:
                    # static_argnums=(5,) marks is_teacher_mode as a compile-time constant.
                    # flax remat's core_fn receives variables as args[0], so user args are offset by 1:
                    # args[1]=x, args[2]=segment_pos, args[3]=cache, args[4]=attn_mask, args[5]=is_teacher_mode
                    blocks.append(flax_nn.remat(TitansBlock, static_argnums=(5,))(
                        **block_kwargs,
                        diff_view=self.config.neural_mem_qkv_receives_diff_view,
                        elastic_net_lambda=self.config.neural_mem_elastic_lambda,
                        huber_loss_delta=self.config.neural_mem_huber_delta,
                        neural_mem_heads=self.config.neural_mem_heads,
                        use_original_attn=False, # Phase 2 uses pure Titans
                    ))
            else:
                if self.config.training_phase !=1:
                    blocks.append(flax_nn.remat(_modules.Block)(**block_kwargs))
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
    def __call__(
        self,
        tokens: Int['*B L'],
        *,
        step: int = 0,
        images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
        positions: Int['*B L_with_mm'] | None = None,
        cache: Optional[Dict[str, Any]] = None,
        attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
        return_hidden_states: bool | None = None,
        **kwargs,
    ) -> Union[DistillationOutput, _transformer.Output]:
        """Forward pass. Broadcasts scalar step to batch dims, then delegates to _forward."""
        # step comes from kontext as scalar (); broadcast to (*B,) so that
        # flatten_unflatten_batch_dim inside _forward sees a properly-shaped array.
        step_b = jnp.broadcast_to(
            jnp.asarray(step, dtype=jnp.int32),
            tokens.shape[:-1],  # (*B,)
        )
        return self._forward(
            tokens,
            step=step_b,
            images=images,
            positions=positions,
            cache=cache,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
            **kwargs,
        )

    @_jax_utils.flatten_unflatten_batch_dim()
    def _forward(
        self,
        tokens: Int['*B L'],
        *,
        step: Int['*B'],
        images: UInt8['*B N H W C'] | UInt8['*B H W C'] | None = None,
        positions: Int['*B L_with_mm'] | None = None,
        cache: Optional[Dict[str, Any]] = None,
        attention_mask: Bool['*B L_with_mm cache_length'] | None = None,
        return_hidden_states: bool | None = None,
        **kwargs,
    ) -> Union[DistillationOutput, _transformer.Output]:
        """Batched forward pass (called by __call__ after step broadcasting)."""
        return_last_only = self.return_last_only

        # MАРКЕР РЕЖИМА: Если есть loss_mask, значит мы в цикле тренировки Kauldron
        is_training = self.config.is_training_mode

        # step has been broadcast to (*B,) and then flattened to (B_flat,) by the
        # decorator; all elements are identical — take [0] to get the scalar step.
        step_scalar = step[0]

        # Evaluate huber delta if it's a schedule
        current_huber_delta = None
        if self.config.neural_mem_huber_delta is not None:
            if callable(self.config.neural_mem_huber_delta):
                current_huber_delta = self.config.neural_mem_huber_delta(step_scalar)
            else:
                current_huber_delta = self.config.neural_mem_huber_delta

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
                is_training=is_training,
                current_huber_delta=current_huber_delta
            )

        if is_training:
            if self.config.training_phase == 2:
                # Phase 2: LM fine-tuning — chunked cross-entropy to avoid materializing
                # the full logit tensor [B, L, 262144] in HBM (~8GB at bf16).
                # Phase 2: checkpointed LM loss — avoids storing [B, L, 262144] logit tensor
                # (~8GB bf16) during backward. jax.checkpoint recomputes logits on-the-fly
                # during backward pass, keeping peak HBM = one forward pass worth of logits.
                # NOTE: no lax.scan here intentionally — scan adds compile-time graph complexity
                # that causes CPU RAM OOM during XLA compilation on TPU v6e-1.
                @jax.checkpoint
                def _lm_loss(hidden, tgt, valid_mask):
                    lc = self.embedder.decode(hidden)
                    if self.config.final_logit_softcap is not None:
                        lc /= self.config.final_logit_softcap
                        lc = jnp.tanh(lc) * self.config.final_logit_softcap
                    # [B, T]
                    ce = optax.softmax_cross_entropy_with_integer_labels(
                        lc.astype(jnp.float32), tgt
                    )
                    # masked mean per sample
                    valid_mask = valid_mask.astype(jnp.float32)
                    denom = jnp.maximum(valid_mask.sum(axis=-1), 1.0)  # [B]
                    loss = (ce * valid_mask).sum(axis=-1) / denom      # [B]
                    pred = jnp.argmax(lc, axis=-1)                     # [B, T]
                    acc = (
                        ((pred == tgt).astype(jnp.float32) * valid_mask).sum(axis=-1)
                        / denom
                    )                                                  # [B]
                    return loss, acc
                tgt = tokens[:, 1:]                    # [B, T]
                valid_mask = inputs.inputs_mask[:, 1:] # [B, T]  <- важно
                lm_loss, lm_acc = _lm_loss(x[:, :-1, :], tgt, valid_mask)
                layer_losses['lm_loss'] = lm_loss
                layer_losses['lm_accuracy'] = lm_acc
                
                return DistillationOutput(
                    logits=jnp.zeros((x.shape[0], 1)),  # логиты не нужны при обучении
                    cache=None if cache is None else new_cache,
                    hidden_states=x if return_hidden_states else None,
                    layer_losses=layer_losses,
                )
            else:
                # Phase 1: distillation — skip logit decoder to save HBM
                return DistillationOutput(
                    logits=jnp.zeros((x.shape[0], 1)),
                    cache=None if cache is None else new_cache,
                    hidden_states=x if return_hidden_states else None,
                    layer_losses=layer_losses,
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
        current_huber_delta: Optional[float] = None,
    ) -> tuple[jax.Array, Dict[str, Any], Dict[str, jax.Array]]:
        x = inputs.embeddings
        # Track previous layer output for hyper-connections (diff view)
        x_prev = x
        
        old_cache = cache or {}
        new_cache = {}
        layer_losses = {}
        
        # Build student mask (truncated window)
        if inputs.attention_mask is not None:
            # Узнаем полную длину накопленного контекста (из маски или кэша)
            k_len = inputs.attention_mask.shape[-1]
            window = 128  # Truncated context for Student
            
            # Используем ИСТИННЫЕ позиции токенов (работает и для префилла L>1, и для декода L=1)
            # inputs.positions имеет форму (batch, seq_len)
            
            # inputs.attention_mask имеет форму (B, L, K_len)
            q_pos = inputs.positions[:, :, None] # [B, L, 1]
            k_pos = jnp.arange(k_len, dtype=jnp.int32)[None, None, :] # [1, 1, K_len]
            
            # Окно смотрит только назад на 'window' токенов
            sliding_window = (q_pos - k_pos) < window
            
            s_mask = inputs.attention_mask & sliding_window
        else:
            s_mask = None

        for i, block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            
            if isinstance(block, TitansBlock):
                if is_training and self.config.training_phase == 1:
                    # PHASE 1: Teacher/Student distillation
                    # 1. Teacher Pass (Full context, original Gemma Attention)
                    layer_cache_teacher, out_teacher = block(
                        x,
                        inputs.positions,
                        old_cache.get(layer_name),
                        inputs.attention_mask, # teacher mask
                        is_teacher_mode=True,
                        kv_seq=x_prev if block.diff_view else None,
                        current_huber_delta=current_huber_delta
                    )

                    # 2. Student Pass (Truncated context, Pure Titans Memory)
                    layer_cache_student, out_student = block(
                        jax.lax.stop_gradient(x),
                        inputs.positions,
                        old_cache.get(layer_name),
                        s_mask, # student mask
                        is_teacher_mode=False,
                        kv_seq=jax.lax.stop_gradient(x_prev) if block.diff_view else None,
                        current_huber_delta=current_huber_delta
                    )

                    # 3. Layer Loss
                    raw_diff = (out_student - jax.lax.stop_gradient(out_teacher)) ** 2
                    layer_loss = jnp.mean(raw_diff, axis=(1, 2), dtype=jnp.float32)
                    layer_losses[f"loss_{layer_name}"] = jnp.log1p(layer_loss)
                    layer_losses[f"raw_mse_{layer_name}"] = layer_loss

                    # 4. Teacher Chain: Update x with Teacher's output to prevent Exposure Bias
                    x_prev = x
                    x = out_teacher
                    
                    # 5. MERGE CACHE: Use Teacher's KV cache (for next layers) 
                    # but Student's updated memory_state (to keep learning)
                    if layer_cache_teacher is not None:
                        merged_cache = dict(layer_cache_teacher)
                        if layer_cache_student is not None and 'memory_state' in layer_cache_student:
                            merged_cache['memory_state'] = layer_cache_student['memory_state']
                        new_cache[layer_name] = merged_cache
                    else:
                        new_cache[layer_name] = None

                else:
                    # PHASE 2 / INFERENCE / EVAL: Pure Titans Memory (Student mode)
                    if i == self.config.titans_phase2_first_layer and is_training:
                        # Optimization for Phase 2 backprop
                        x = jax.lax.stop_gradient(x)
                        x_prev = jax.lax.stop_gradient(x_prev)
                        
                    layer_cache_student, out_student = block(
                        x,
                        inputs.positions,
                        old_cache.get(layer_name),
                        s_mask if s_mask is not None else inputs.attention_mask,
                        False,  # is_teacher_mode
                        x_prev if block.diff_view else None,
                        current_huber_delta=current_huber_delta
                    )
                    x_prev = x
                    x = out_student
                    new_cache[layer_name] = layer_cache_student
            else:
                # Standard Gemma Block
                layer_cache, out_next = block(
                    x,
                    inputs.positions,
                    old_cache.get(layer_name),
                    inputs.attention_mask,
                )
                x_prev = x
                x = out_next
                new_cache[layer_name] = layer_cache
            
        x = self.final_norm(x)
        
        # Add total distillation loss to satisfy Kauldron trainer expectations in Phase 1
        if self.config.training_phase == 1 and is_training:
            total_distill = jnp.zeros((), dtype=jnp.float32)
            count = 0
            for k, v in layer_losses.items():
                if k.startswith("loss_layer_"):
                    total_distill += jnp.mean(v)
                    count += 1
            if count > 0:
                # Транслируем скалярный лосс на размер батча, чтобы декоратор мог его развернуть
                layer_losses['lm_loss'] = jnp.broadcast_to(total_distill / count, (x.shape[0],))
            else:
                layer_losses['lm_loss'] = jnp.zeros((x.shape[0],), dtype=jnp.float32)

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
            
            if i in self.config.titans_layer_indices and i >= self.config.titans_phase2_first_layer:
                mem_state = init_memory_state(
                    batch_size=batch_size,
                    dim=self.config.embed_dim,
                    heads=self.config.neural_mem_heads,
                    dim_head=256,
                    dtype=dtype
                )
                attn_cache['memory_state'] = mem_state
            cache[layer_name] = attn_cache
            
        return cache
