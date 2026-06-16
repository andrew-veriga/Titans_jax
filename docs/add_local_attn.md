# Архитектурные изменения: Гибрид LOCAL Attention + Neural Memory

> **Дата:** 2026-06-15  
> **Статус:** Внедрено и протестировано (4/4 decode tests ✅, 4/4 tree utils tests ✅)

---

## Краткое содержание

Внедрена гибридная архитектура **MAG (Memory as Gate)** с **LOCAL sliding window attention** (window=128) как дополнение к Neural Long-Term Memory (NLM). Параллельно исправлен критический баг decode-режима (возврат нулей вместо мусора) и добавлена буферизация токенов. Ниже — детальное описание каждого изменения, его причин и теоретических оснований.

---

## 1. Гибрид LOCAL Attention + Neural Memory (MAG-архитектура)

### Что изменилось

**Было** (чистый Titans Memory):
```python
# TitansBlock.__call__ — student/inference mode
retrieved, next_mem_state, avg_mem_loss = self.memory(...)
combined_output = retrieved  # только память, без attention
```

**Стало** (гибрид MAG):
```python
# LOCAL sliding window attention: точный near-context (window=128)
new_attn_cache, local_attn_output = self.local_attn(
    inputs_normalized, segment_pos, cache, attn_mask,
)
# Neural Memory: far-context
retrieved, next_mem_state, avg_mem_loss = self.memory(...)

# Динамический гейт: балансировка local_attn vs memory
gate = jax.nn.sigmoid(jnp.clip(self.memory_gate_proj(inputs_normalized), -10.0, 10.0))

# HYBRID: local_attn (near) + gate * memory (far)
combined_output = local_attn_output + gate * retrieved
```

### Почему

**Проблема чистого NLM:** Neural Memory обновляется чанками (`chunk_size=32`). В decode-режиме (по 1 токену) память **не возвращает ничего полезного** — нет чанка для обработки. Модель теряет весь локальный контекст между обновлениями памяти.

**Решение:** добавить LOCAL sliding window attention (window=128), который обеспечивает:
- **Near-context** (точное внимание к 128 ближайшим токенам) — комплементарно к NLM
- **Гарантированный сигнал** при decode — `local_attn` всегда активен, даже когда memory возвращает нули
- **Разделение ответственности:** local_attn = точное краткосрочное, memory = сжатое долгосрочное

### Теоретическое обоснование

Это вариация архитектуры **MAG (Memory as Gate)** из оригинальной статьи Titans:

> **MAG — Memory as Gate:** Attention и NLM работают **параллельно**. Выходы объединяются через обучаемый механизм гейтирования. Модель динамически балансирует локальную точность и глобальный контекст.
>
> — [[Memory as Context (MAC)]] → раздел "Три варианта интеграции"

Оригинальная статья Titans описывает три варианта:
| Вариант | Интеграция | Применение |
|---------|-----------|------------|
| MAC | Конкатенация токенов памяти с входом | Сверхдлинный контекст |
| **MAG** | **Параллель + гейт** ← наш выбор | **Динамические задачи** |
| MAL | Последовательное стекирование | Иерархические задачи |

Наш вариант отличается от канонического MAG тем, что вместо **глобального** attention используем **локальный** (sliding window=128). Это сделано по двум причинам:

1. **Эффективность:** Global attention имеет $O(L^2)$ сложность, что непрактично для длинных контекстов. Local sliding — $O(L \cdot w)$ где $w=128$.
2. **Дополнение, не дублирование:** NLM уже хранит far-context в весах MLP. Local attention покрывает gap между "совсем недавним" (KV-cache decode) и "далёким" (memory).

> **Из систематического обзора Memory-Augmented Transformers (Huawei):**
> "Emerging solutions: иерархическое буферирование, surprise-gated updates"
>
> — [[Memory-Augmented Transformers — Systematic Review]]

Гибрид local_attn + memory реализует именно эту идею: attention работает как "рабочая память" (working memory), а NLM — как "долговременная" (long-term), связываемая через gate.

### Реализация в коде

**`gemma_titans.py`, `TitansBlock.setup()`:**
```python
# LOCAL sliding window attention (always created)
self.local_attn = _modules.Attention(
    num_heads=self.num_heads,
    features=self.embed_dim,
    head_dim=self.head_dim,
    num_kv_heads=self.num_kv_heads,
    attn_type=_modules.AttentionType.LOCAL_SLIDING,
    sliding_window_size=self.sliding_window_size,  # 128
    ...
)
```

**`gemma_titans.py`, `Gemma3_1B_Titans._apply_attention()`:**
Student mask строится на основе sliding window:
```python
window = 128  # Truncated context for Student
q_pos = inputs.positions[:, :, None]       # [B, L, 1]
k_pos = jnp.arange(k_len)[None, None, :]   # [1, 1, K_len]
sliding_window = (q_pos - k_pos) < window
s_mask = inputs.attention_mask & sliding_window
```

---

## 2. Decode Fix: нули вместо мусора + буферизация токенов

### Что изменилось

**Было** (`titans.py`, `NeuralMemory.__call__`):
```python
def __call__(self, seq, memory_state=None, ...):
    # ... вызов retrieve_memories ...
    return values  # всегда, даже при seq_len=1
```

При decode (`seq_len=1 < chunk_size=32`), функция всё равно возвращала результат `retrieve_memories`, который падал в ветку "pad with empty memory embed":
```python
# retrieve_memories, строки 598-601:
empty_embeds = repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=self.chunk_size-1)
values = jnp.concatenate([empty_embeds, values], axis=1)
```

`empty_memory_embed` — это **обучаемый параметр** (форма `(dim,)`, init `normal(stddev=0.02)`). Он возвращался для **каждого** decode-шага, искажая сигнал:
```
output = gate * empty_memory_embed + x  ≈  0.88 * МУСОР + x
```

**Стало** (`titans.py`, `NeuralMemory.__call__`, строки 614-652):
```python
if seq_len < self.chunk_size:
    # Buffer tokens; return zeros; no retrieval during decode
    past_weights, past_momentum, token_buffer, buffer_count = memory_state
    new_buffer = jax.lax.dynamic_update_slice(token_buffer, seq, (0, buffer_count, 0))
    new_count = buffer_count + seq_len

    new_memory_state = jax.lax.cond(
        new_count >= self.chunk_size,
        _store_and_reset,   # buffer full → store_memories + reset
        _keep_buffer,       # keep accumulating
        operand=None,
    )
    ret = jnp.zeros((batch, seq_len, self.dim), dtype=seq.dtype)  # ВСЕГДА нули
    return ret, new_memory_state
```

### Почему

1. **Нули вместо мусора:** При decode `gate * 0 + local_attn = local_attn` — чистый near-context, без шума. Сигнал не искажается.

2. **Буферизация:** Накапливаем decode-токены в `token_buffer` (форма `(batch, chunk_size, dim)`). Когда буфер заполняется (достигает `chunk_size`), вызываем `store_memories` для обновления NLM и сбрасываем буфер. Это гарантирует, что:
   - Память обновляется правильными чанками (32 токена)
   - Между обновлениями decode работает корректно (нули + local_attn)
   - KV-cache и memory_state эволюционируют независимо

### Теоретическое обоснование

**Surprise-gated learning:** NLM обучается через градиентный спуск на чанках. Из [[Нейронная долгосрочная память (NLM)]]:

> Для каждого входного токена $x_t$ вычисляются проекции $k_t$ (ключ) и $v_t$ (значение). Память обновляется через градиентный спуск на ассоциативной loss.
>
> $M_t = M_{t-1} + S_t$
>
> Где $S_t$ — накопленный «сюрприз» с моментом.

Обновление по одному токену за раз дало бы:
- Зашумлённые градиенты (1 пример вместо чанка)
- Несогласованные метрики loss
- $32\times$ больше вызовов `value_and_grad` при decode

Буферизация гарантирует, что память обновляется чанками, как при training/prefill.

### Реализация

**`titans.py`, `init_memory_state()` (строки 114-142):**
```python
# Token buffer: pre-allocated (batch, chunk_size, dim) for JAX static shapes
token_buffer = jnp.zeros((batch_size, chunk_size, dim), dtype=dtype)
buffer_count = jnp.int32(0)
return (initial_weights, momentum, token_buffer, buffer_count)  # 4-element tuple
```

**`titans.py`, `store_memories()` (строка 370):**
```python
# memory_state is now 4-element: (weights, momentum, token_buffer, buffer_count)
# store_memories only needs weights and momentum
past_weights, past_momentum = past_state[:2]
```

**Тест `tests/test_titans_decode.py`** проверяет:
- `init_memory_state` возвращает 4-элементный кортеж ✅
- Decode (seq_len=1) возвращает нули ✅
- Буферизация накапливает токены (buffer_count 1→2→3→0) ✅
- Prefill (seq_len ≥ chunk_size) работает как раньше ✅

---

## 3. `stop_gradient(local_attn)` в Phase 1

### Что изменилось

**`gemma_titans.py`, `TitansBlock` (строки 64-66, 209-212):**
```python
class TitansBlock(_modules.Block):
    freeze_local_attn: bool = False  # Phase 1: stop_gradient on local_attn

    def __call__(self, ...):
        ...
        if self.freeze_local_attn:
            local_attn_output = jax.lax.stop_gradient(local_attn_output)
```

В `Gemma3_1B_Titans.setup()` (строки 384-390):
```python
if self.config.training_phase == 1:
    blocks.append(TitansBlock(
        **block_kwargs,
        use_original_attn=True,
        freeze_local_attn=True,  # Phase 1: stop_gradient on local_attn
    ))
```

### Почему

В Phase 1 (послойная дистилляция) цель — обучить **память** имитировать поведение teacher (Gemma с global attention). Если `local_attn` тоже обучается, возникает **identity shortcut**: модель может проигнорировать память и просто обучить `local_attn` ≈ `attn` (teacher).

`stop_gradient(local_attn)` принудительно делает память **единственным обучаемым путём** в attention-ветви. Это реализует принцип **surprise-gated updates**: градиенты от distillation loss идут только через memory, заставляя NLM учиться.

### Теоретическое обоснование

> **Surprise-gated updates:** Приоритетное запоминание через градиент loss как меру новизны.
>
> — [[AI Papers Academy — Titans]], [[Shaped.ai — Titans: Neural Memory Systems]]

Если local_attn обучается одновременно с памятью, "сюрприз" (gradient) распределяется между двумя путями. `stop_gradient` концентрирует весь surprise в памяти, что соответствует принципу "memorization without overfitting through surprise-based learning" из оригинальной статьи.

Дополнительно, из систематического обзора:
> "Emerging solutions: surprise-gated updates" — [[Memory-Augmented Transformers — Systematic Review]]

`stop_gradient` — это архитектурная реализация gating'а градиентного потока: в Phase 1 gate "закрыт" для local_attn, "открыт" для memory.

---

## 4. Чекпойнты: `local_attn` в `titans_tree_utils.py`

### Что изменилось

**`titans_tree_utils.py`, `_TITANS_KEYS` (строки 36-40):**
```python
_TITANS_KEYS = frozenset({
    'memory', 'memory_gate_proj',
    'titans_ffn', 'titans_pre_ffw_norm', 'titans_post_ffw_norm',
    'local_attn',  # LOCAL sliding attention — Titans hybrid component
})
```

**Auto-init из `attn` при загрузке (строки 135-136):**
```python
if 'local_attn' not in layer_params and 'attn' in layer_params:
    layer_params['local_attn'] = copy.deepcopy(layer_params['attn'])
```

### Почему

При загрузке старых чекпойнтов (Phase 1/2 без `local_attn`) необходимо:
1. Идентифицировать `local_attn` как Titans-компонент (не Gemma) → добавлен в `_TITANS_KEYS`
2. Инициализировать `local_attn` из предобученных весов `attn` (teacher attention), а не случайно → auto-init в `merge_titans_params`
3. Сохранить `local_attn` при `remove_dead_attn=True` → удаляется только `attn`, не `local_attn`

### Теоретическое обоснование

Transfer learning: `local_attn` (sliding window) и `attn` (global) имеют одинаковую архитектуру Attention, отличаясь только типом маски. Инициализация из `attn` даёт хороший starting point — предобученные Q/K/V проекции работают и для локального окна.

Тест `tests/test_tree_utils.py` проверяет все 4 сценария:
- `local_attn` ∈ `_TITANS_KEYS` ✅
- `split_titans_params` помещает `local_attn` в titans_tree ✅
- `merge` с `remove_dead_attn` сохраняет `local_attn`, удаляет `attn` ✅
- `merge` auto-init `local_attn` из `attn` когда отсутствует ✅

---

## 5. Динамический гейт: `memory_gate_proj` (Dense layer)

### Что изменилось

**Было** (статический вектор):
```python
self.memory_gate = self.param('memory_gate', nn.initializers.constant(0.5), (self.embed_dim,))
gate = jax.nn.sigmoid(self.memory_gate)
```

**Стало** (динамический Dense-слой):
```python
self.memory_gate_proj = flax_nn.Dense(
    features=self.embed_dim,
    use_bias=True,
    kernel_init=flax_nn.initializers.lecun_normal(),
    bias_init=flax_nn.initializers.constant(2.0),  # sigmoid(2) ≈ 0.88
    name='memory_gate_proj'
)
gate = jax.nn.sigmoid(jnp.clip(self.memory_gate_proj(inputs_normalized), -10.0, 10.0))
```

### Почему

1. **Контекстозависимость:** Статический вектор даёт одинаковый gate для всех токенов. Dense-слой вычисляет gate на основе текущего токена — модель учится когда доверять памяти, а когда local_attn.

2. **`bias_init=2.0`:** `sigmoid(2) ≈ 0.88` — gate изначально "открыт" для памяти. Это даёт памяти шанс быть услышанной на ранних этапах обучения.

3. **`clip(-10, 10)`:** Предотвращает насыщение sigmoid в зонах с нулевым градиентом.

### Миграция старых чекпойнтов

**`titans_tree_utils.py`, `migrate_static_gate_to_dynamic()`** удаляет старый ключ `memory_gate`:
```python
def migrate_static_gate_to_dynamic(params):
    # Removes old static 'memory_gate', new 'memory_gate_proj' inits randomly
```

---

## Сводная таблица изменений

| # | Изменение | Файл | Обоснование |
|---|-----------|------|-------------|
| 1 | Гибрид MAG: `local_attn` + `gate * memory` | `gemma_titans.py` | [[Memory as Context (MAC)]] → MAG вариант |
| 2 | Decode: нули + буферизация | `titans.py` | Surprise-gated learning, чанковое обновление |
| 3 | `stop_gradient(local_attn)` Phase 1 | `gemma_titans.py` | Surprise-gated updates principle |
| 4 | `local_attn` в `_TITANS_KEYS` + auto-init | `titans_tree_utils.py` | Совместимость чекпойнтов |
| 5 | Динамический `memory_gate_proj` | `gemma_titans.py` | Контекстозависимый gating |

---

## Аффектированные файлы

- **`gemma_titans.py`** — TitansBlock: local_attn, gate, freeze_local_attn; _apply_attention: student mask
- **`titans.py`** — init_memory_state (4-element tuple), NeuralMemory.__call__ (decode zeros + buffer)
- **`titans_tree_utils.py`** — _TITANS_KEYS += local_attn; merge auto-init; migrate_static_gate
- **`routing_optimizer.py`** — `local_attn` добавлен в `partial_updates` select-mask (критический фикс; без него градиенты local_attn не обновлялись бы)
- **`tests/test_titans_decode.py`** — 4 теста decode-режима
- **`tests/test_tree_utils.py`** — 4 теста tree utils с local_attn
- **`titans_ckpts.py`** — `SkipTitans.transform()`: после merge, копирует `attn`→`local_attn` для Titans-слоёв (критический фикс для FIRST_RUN=True, когда `local_attn` инициализируется случайно)
- **`colabs/Titans_jax_Phase1_training.ipynb`** — `_ensure_local_attn()` helper + вызов в `FullParamsInit.transform()`: копирует `attn`→`local_attn` для старых чекпойнтов при FIRST_RUN=False
- **`colabs/Titans_jax_Phase2_training.ipynb`** — `adaptive_max_lr` 5e-4 → 3e-4; `lr_adam` 5e-4 → 2e-4 (local_attn берёт часть near-context, поэтому LR снижены)

---

## Ссылки на wiki

- [[Memory as Context (MAC)]] — три варианта интеграции (MAC, MAG, MAL)
- [[Нейронная долгосрочная память (NLM)]] — принцип работы NLM, store/retrieve
- [[AI Papers Academy — Titans]] — surprise-gated learning
- [[Shaped.ai — Titans: Neural Memory Systems]] — MAG: gating между памятью и вниманием
- [[Memory-Augmented Transformers — Systematic Review]] — иерархическое буферирование, surprise-gated updates
- [[Вентиль удержания (Retention Gate)]] — decay factor, forget/retention gates
- [[Чанкинг и параллельное обучение]] — почему память обновляется чанками