# План восстановления конвергенции Phase 1

## Точный диагноз: Plateau loss=0.5 — утечка градиента в `titans_ffn`

### Корневая причина

Сравните student path в двух версиях:

**d3b801e (рабочая):** Память — **единственный** обучаемый путь
```python
combined = gate * tanh(retrieved)       # только память
outputs = self.mlp(combined)            # mlp из Gemma, ЗАМОРОЖЕН (нет в partial_updates)
```

**main (сломанная):** Два конкурирующих обучаемых пути
```python
combined = local_attn + gate * retrieved   # память
outputs = self.titans_ffn(combined)         # ОТДЕЛЬНЫЙ FFN, ОБУЧАЕМЫЙ (есть в partial_updates!)
```

В `routing_optimizer.py:157-162` `titans_ffn` добавлен в `partial_updates`:
```python
mask=kd.optim.select([
    "memory", "memory_gate_proj", "local_attn",
    "titans_ffn", "titans_pre_ffw_norm", "titans_post_ffw_norm",  # ← ОБУЧАЕМЫЙ!
])
```

**Что происходит:** `titans_ffn` (warm-started из Gemma `mlp`) — это гораздо более мощный и лёгкий для оптимизатора путь, чем нейронная память (которая обновляется через внутренний gradient descent). Модель сводит loss, обучая `titans_ffn` компенсировать разницу `(local_attn - attn)`, а `gate*retrieved` остаётся на стартовом уровне → **loss plateau**.

На шаге 0: `titans_ffn = mlp` (из Gemma), `local_attn ≈ attn`, поэтому `delta ≈ gate*retrieved` → MSE ≈ 0.5 (как наблюдается). Затем `titans_ffn` обучается компенсировать `(local_attn - attn)`, loss слегка колеблется, но память не учится.

### Что НЕ нужно менять (исключено после анализа)

- ❌ **RoPE для `local_attn`** — эмпирически доказано пользователем: оба варианта (1M и 10K) не восстановили конвергенцию. RoPE не причина plateau. Оставляем `rope_base_frequency=self.rope_base_frequency`.
  - Доп. аргумент: Q/K/V веса teacher attention обучены с `global_base_frequency=1M`. Замена на 10K рассогласует обученные веса с позиционным кодированием.
- ❌ **`optimizer_mask_plan.md` (regex wildcards)** — маска `kd.optim.select` работает корректно. Я проверил исходник `_make_regex`: для `"memory"` создаётся regex `(?:^|\.)memory(?:$|\.)`, который через `re.search` матчит путь `layer_23.memory.memory_model.weight_0`. Все ключи (`memory`, `local_attn`, `titans_ffn`, `memory_gate_proj`) корректно попадают в обучаемую маску.
- ❌ **Dynamic init через `jax.lax.cond`** в `titans.py:631-638` — вторичен. При фиксе 1 градиент к `memory_model` не нужен, т.к. MemoryMLP — это **fast weights** (обновляются через inner gradient descent внутри `store_memories`, а не через внешний оптимизатор).
- ❌ **`init_memory_state` с `PRNGKey(0)` vs dynamic** — в Phase 1 не критично, т.к. MemoryMLP — fast weights. Внешний оптимизатор обучает только проекций (`to_queries`, `to_keys_values`, `combine_heads`).

---

## Таблица фаз: что обучается

| | Phase 1 (дистилляция) | Phase 2 (LM fine-tuning) | Инференс (генерация) |
|---|---|---|---|
| **FFN** | `mlp` из Gemma (**заморожен**) | `titans_ffn` (**обучаемый**, warm-started из `mlp`) | `titans_ffn` (**заморожен**) |
| **Memory projections** (`to_queries`, `to_keys_values`, `combine_heads`) | **обучаемые** | **обучаемые** | **заморожены** |
| **memory_gate_proj** | **обучаемый** | **обучаемый** | **заморожен** |
| **local_attn** | заморожен (`stop_gradient`) | **обучаемый** | **заморожен** |
| **MemoryMLP** (fast weights) | обновляется через inner GD | обновляется через inner GD | **обновляется через inner GD** ← "учится на лету" |

**Обоснование НЕ замораживать memory projections в Phase 2:** Проекции определяют, *как* память извлекает контекст. В Phase 2 модель видит реальный LM-сигнал, и проекции должны донастраиваться. `titans_ffn` не будет доминировать, т.к. memory обучена в Phase 1 и даёт осмысленный сигнал с первого шага.

---

## План точечных фиксов (5 фиксов)

| # | Фикс | Файл | Приоритет | Обоснование |
|---|------|------|-----------|-------------|
| **1** | **Phase 1: student → teacher's `mlp`/`pre_ffw_norm`/`post_ffw_norm` (замороженные), а НЕ `titans_ffn`** + условное создание `titans_ffn` только при `use_original_attn=False` | `gemma_titans.py` | 🔴 **Критический** | Убирает утечку градиента; память становится единственным обучаемым путём (как в d3b801e) |
| **2** | **Loss: `cos_by_softmax` вместо `distill_mse`** | `gemma_titans.py:655` | 🟡 Высокий | В d3b801e работала именно `cos_by_softmax` |
| **3** | **`lr_adam`: 3e-4 → 1e-4** | `colabs/Titans_jax_Phase1_training.ipynb` | 🟡 Средний | Вернуть рабочий LR из d3b801e |
| **4** | **Вернуть `tanh(retrieved)` ограничение** | `gemma_titans.py:232` | 🟢 Низкий | Стабилизация `gate*retrieved` |
| **5** | **Инициализация `titans_ffn ← mlp` для Phase 2** (исправить порядок в `titans_ckpts.py`) | `titans_ckpts.py` | 🟡 Высокий | Warm-start `titans_ffn` из Gemma `mlp` при `FIRST_RUN=True` Phase 2 |

---

### Детали Фикса 1 (главного)

В `TitansBlock`, использовать `self.use_original_attn` как индикатор Phase 1 (он `True` только в Phase 1, см. `Gemma3_1B_Titans.setup()` строки 388-394).

**`gemma_titans.py`, `TitansBlock.setup()`:** создавать `mlp`/`pre_ffw_norm` только при `use_original_attn=True` (уже есть), а `titans_ffn`/`titans_pre_ffw_norm`/`titans_post_ffw_norm` — только при `use_original_attn=False`.

```python
# В setup():
if self.use_original_attn:
    # Phase 1: teacher и student используют Gemma FFN (замороженный)
    self.pre_ffw_norm = _layers.RMSNorm()
    self.mlp = _modules.FeedForward(...)
    if self.use_post_ffw_norm:
        self.post_ffw_norm = _layers.RMSNorm()
else:
    # Phase 2/Inference: student использует titans_ffn (обучаемый в Phase 2)
    self.titans_pre_ffw_norm = _layers.RMSNorm()
    self.titans_ffn = _modules.FeedForward(...)
    if self.use_post_ffw_norm:
        self.titans_post_ffw_norm = _layers.RMSNorm()
```

**`gemma_titans.py`, `TitansBlock.__call__()`:**
```python
# Phase 1 (use_original_attn=True): и teacher, и student → Gemma mlp (заморожен)
# Phase 2/Inference (use_original_attn=False): student → titans_ffn
if self.use_original_attn:
    outputs = self.pre_ffw_norm(combined_output)
    outputs = self.mlp(outputs)
    if self.post_ffw_norm is not None:
        outputs = self.post_ffw_norm(outputs)
else:
    outputs = self.titans_pre_ffw_norm(combined_output)
    outputs = self.titans_ffn(outputs)
    if self.titans_post_ffw_norm is not None:
        outputs = self.titans_post_ffw_norm(outputs)
```

Поскольку `mlp`/`pre_ffw_norm`/`post_ffw_norm` **не в** `partial_updates`, они заморожены → единственный обучаемый путь в student — память. Точно как в d3b801e.

---

### Детали Фикса 5 (инициализация для Phase 2)

В `titans_ckpts.py`, `SkipTitans.transform()`, копирование `titans_ffn ← mlp` должно происходить **до** cleanup (когда `mlp` ещё доступен из Gemma checkpoint).

**Текущий баг (порядок):**
1. `merge_titans_params` — добавляет `titans_ffn` (random)
2. Cleanup — удаляет `mlp` из Titans слоёв
3. Копирование `titans_ffn ← mlp` — **FAIL**: `mlp` уже удалён

**Исправленный порядок:**
1. `merge_titans_params` — добавляет `titans_ffn` (random)
2. **Копирование `titans_ffn ← mlp`** (mlp ещё доступен из Gemma)
3. Cleanup — удаляет `mlp` (больше не нужен)

---

### Детали Фиксов 2, 3, 4

**Фикс 2 (`gemma_titans.py:655`):**
```python
# Было: layer_loss = self.distill_mse(delta_teacher, delta_student)
# Стало:
layer_loss = self.cos_by_softmax(delta_teacher, delta_student)  # (B, L)
```

**Фикс 3 (`colabs/Titans_jax_Phase1_training.ipynb`, opt_params):**
```python
# Было: "lr_adam": ... init_value=3e-4, peak_value=3e-4 ...
# Стало: "lr_adam": ... init_value=1e-4, peak_value=1e-4 ...
```

**Фикс 4 (`gemma_titans.py`, student path):**
```python
# Было: combined_output = local_attn_output + gate * retrieved
# Стало:
retrieved = jnp.tanh(retrieved)  # bounds to [-1, 1]
combined_output = local_attn_output + gate * retrieved
```

---

## Порядок применения

1. Применить **все 5 фиксов**
2. Запустить Phase 1 обучение (500 шагов) → проверить, что loss снижается
3. Если конвергенция восстановлена → проверить Phase 2 (с `titans_ffn` warm-started)

## Риски

- **Phase 1: неиспользуемые параметры.** `titans_ffn` создаётся только при `use_original_attn=False` → в Phase 1 он не создаётся вообще. Обратное: `mlp` создаётся только при `use_original_attn=True` → в Phase 2 не создаётся. Это решает проблему unused parameters.
- **Phase 2 чекпойнты.** После Phase 1 чекпойнт содержит `mlp`/`pre_ffw_norm` (замороженные). При переходе в Phase 2 они не нужны (Phase 2 использует `titans_ffn`). Нужно убедиться, что `merge_titans_params` корректно обрабатывает отсутствие `mlp`.