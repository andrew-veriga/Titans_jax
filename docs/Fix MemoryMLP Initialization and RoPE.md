# Plan: Fix MemoryMLP Initialization, RoPE, and Phase 1 Init

## Background & Motivation

We have identified three distinct, critical bugs preventing the model from learning correctly across different phases. We need to fix all of them to unblock the training process.

1.  **Phase 1 Initialization Bug (`titans_ckpts.py`)**: When `FIRST_RUN=True`, the `SkipTitans` loader attempts to warm-start `local_attn` and `titans_ffn` from pretrained Gemma weights. However, it fails to `unfreeze` the Flax `FrozenDict` before attempting to check its type and modify it. This causes the initialization block to be silently skipped, leaving Phase 1 to start with a completely random FFN and local attention, preventing the loss from converging.
2.  **MemoryMLP Initialization Bug (`titans.py`)**: The `init_memory_state` function generates fresh random weights using a fixed seed (`jax.random.PRNGKey(0)`) for every batch. This means the actual trainable parameters of the `MemoryMLP` inside the `NeuralMemory` module (which are updated by the optimizer) are entirely ignored. Every forward pass starts with the same random memory weights, making the optimizer's updates useless.
3.  **RoPE Frequency Mismatch (`gemma_titans.py`)**: For layers that are "global" in the original Gemma architecture (e.g., layer 23), the `TitansBlock` receives the `global_base_frequency` (1,000,000). The `local_attn` module (which is a sliding window attention) incorrectly uses this global frequency instead of the intended `local_base_frequency` (10,000). This extremely high frequency on a small window (128) makes the positional encoding meaningless, degrading the local attention to a simple average pooling.

## Objective

Apply all three fixes simultaneously to ensure the architecture functions as intended across all phases.

## Implementation Steps

### Step 1: Fix Phase 1 `unfreeze` bug (`titans_ckpts.py`)
In `SkipTitans.transform`, we need to properly unfreeze `state.params` before modifying it, and then freeze it back.

*Code change in `titans_ckpts.py`:*
```python
    from flax.core import unfreeze, freeze
    loaded_params = unfreeze(state.params)
    _TITANS_INIT_MAP = {
        'local_attn': 'attn',
        'titans_ffn': 'mlp',
        'titans_pre_ffw_norm': 'pre_ffw_norm',
        'titans_post_ffw_norm': 'post_ffw_norm',
    }
    for key, layer_params in loaded_params.items():
      if 'layer_' in key and isinstance(layer_params, dict):
        is_titans = 'memory' in layer_params or 'memory_gate_proj' in layer_params
        if is_titans:
          for titans_key, gemma_key in _TITANS_INIT_MAP.items():
            if titans_key in layer_params and gemma_key in layer_params:
              layer_params[titans_key] = copy.deepcopy(layer_params[gemma_key])
    state = state.replace(params=freeze(loaded_params))
```

### Step 2: Fix MemoryMLP Initialization (`titans.py`)

In `NeuralMemory.__call__`, we will stop generating random weights. Instead, we will fetch the base parameters directly from `self.memory_model` and broadcast them. We must also update or remove the usage of the old `init_memory_state`.

*Code changes in `titans.py`:*
1. Remove or modify `init_memory_state` to only return the `token_buffer` and `buffer_count`.
2. Inside `NeuralMemory.__call__`, when `memory_state` is missing, fetch weights from `self.variables['params']['memory_model']`.

```python
        if not exists(memory_state):
            # Force parameter initialization by passing a dummy input
            dummy_input = jnp.zeros((1, self.dim_head), dtype=seq.dtype)
            _ = self.memory_model(dummy_input)
            
            base_params = self.variables['params']['memory_model']
            
            initial_weights = {}
            for i in range(self.mlp_depth):
                w = base_params[f'weight_{i}'] 
                w_bcast = jnp.broadcast_to(w[None, None, ...], (batch, self.heads, self.dim_head, self.dim_head))
                initial_weights[f'weight_{i}'] = w_bcast
                
            momentum = jax.tree_util.tree_map(jnp.zeros_like, initial_weights)
            token_buffer = jnp.zeros((batch, self.chunk_size, self.dim), dtype=seq.dtype)
            buffer_count = jnp.int32(0)
            
            memory_state = (initial_weights, momentum, token_buffer, buffer_count)
```

### Step 3: Fix RoPE for `local_attn` (`gemma_titans.py`)

In `gemma_titans.py`, inside `TitansBlock.setup`, hardcode the `local_attn` module to use the local RoPE frequency.

*Code change in `gemma_titans.py`:*
```python
        self.local_attn = _modules.Attention(
            num_heads=self.num_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            attn_type=_modules.AttentionType.LOCAL_SLIDING,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            rope_base_frequency=10000.0,  # FORCE LOCAL ROPE
            rope_scale_factor=1.0,        # FORCE LOCAL ROPE
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            sliding_window_size=self.sliding_window_size,
            use_qk_norm=self.use_qk_norm,
        )
```

## Verification
1. Verify `titans_ckpts.py` applies the warm-start logic without skipping it.
2. Verify that `titans.py` correctly extracts base weights from `self.memory_model` instead of using `PRNGKey(0)`.
3. Verify that `gemma_titans.py` hardcodes `10000.0` for `local_attn` RoPE frequency.


Эти три исправления кардинально меняют математику и физику процесса обучения. Вот как конкретно каждое из них  помогает лоссу начать снижаться:

  ---

  1. Фикс unfreeze (Фаза 1): Устранение неустранимого шума
   * Как было: Из-за пропуска инициализации в SkipTitans слой local_attn стартовал со случайными весами. В Фазе     1 на local_attn наложен stop_gradient(). Это значит, что случайный шум от локального внимания подмешивался     в скрытое состояние напрямую, и оптимизатор физически не мог его обучить или уменьшить. Памяти приходилось     пытаться "компенсировать" гигантский шум случайных весов в размерности $D=1152$, что приводило к коллапсу
     градиентов.
   * Как стало: Теперь local_attn и titans_ffn инициализируются идеальными весами Gemma. Модель на шаге 0
     выдает осмысленный текст (лосс сразу низкий). Памяти больше не нужно бороться с хаосом — ей нужно лишь
     научиться плавно заменять далекий контекст, что является простой задачей дистилляции.

  ---

  2. Фикс инициализации MemoryMLP (Фазы 2 и 3): Подключение оптимизатора к сети
   * Как было: Оптимизатор честно считал градиенты для параметров MemoryMLP (весов нейронной памяти) и обновлял     их в PyTree. Но в forward-проходе модель их никогда не читала! Вместо этого при каждом шаге использовались     мертвые случайные веса, сгенерированные по ключу PRNGKey(0). Обучение базовых параметров памяти шло "в
     никуда", а модель каждый раз пыталась учиться с нуля на одном и том же случайном шуме.
   * Как стало: Теперь при старте батча (или prefill-шага) модель берет реальные веса модели, которые только
     что обновил оптимизатор. Градиентная связь замкнулась: оптимизатор обновил веса → модель прочитала
     обновленные веса на следующем шаге → лосс начал снижаться.

  ---

  3. Фикс RoPE частоты: Восстановление пространственного зрения модели
   * Как было: На 23-м слое (глобальном слое Gemma) локальное внимание local_attn (окно 128) использовало
     глобальную частоту RoPE 1,000,000. Из-за этого угловое смещение позиционных эмбеддингов для соседних
     токенов было микроскопическим (≈ 10⁻⁶ радиан). Для внимания позиции токенов внутри окна становились
     неразличимы. Внимание вырождалось в хаотичное усреднение (модель не понимала, какой токен идет за каким),
     что ломало локальный синтаксис и мешало снижению лосса.
   * Как стало: Теперь локальное внимание всегда работает на локальной частоте 10,000. Модель снова четко
     "видит" порядок слов внутри скользящего окна 128.

  ---

  Общий итог:
  Вместо того чтобы бороться с тремя непреодолимыми математическими барьерами (замороженным шумом, мертвым
  градиентным путем к памяти и слепым позиционным кодированием), модель теперь находится в идеальной стартовой
  точке:
   1. Residual stream полностью сохранен и идентичен Gemma.
   2. Внимание отлично видит локальный порядок слов.
   3. Оптимизатор напрямую управляет базовыми весами нейропамяти.