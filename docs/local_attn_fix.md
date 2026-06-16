## Анализ влияния `local_attn` на обучение и предложение изменений

После изучения всех конфигов (`experimental_config`, `opt_params`, `routing_optimizer.py`) я обнаружил **один критический баг** и несколько важных моментов.

---

### 🔴 Критический баг: `local_attn` отсутствует в `partial_updates`

В `routing_optimizer.py` (строки 157-161) список `partial_updates` не содержит `local_attn`:
```python
mask=kd.optim.select([
    "memory", "memory_gate_proj", "titans_ffn", "titans_pre_ffw_norm", "titans_post_ffw_norm",
    "layer_24", "layer_25", "final_norm"
    # ← local_attn ОТСУТСТВУЕТ!
]),
```

**Последствие:** В Phase 2 `local_attn` **эффективно заморожен** — XLA не строит градиенты для него, даже если архитектурно `stop_gradient` убран. Модель не может обучать локальное внимание.

**Фикс:** Добавить `"local_attn"` в select-список.

---

### 🟡 Проблема инициализации `local_attn` в Phase 1

При `FIRST_RUN=True` в Phase 1, `local_attn` инициализируется **случайно** (через SkipTitans). Хотя он заморожен (`freeze_local_attn=True`), его случайный выход добавляется в student: `combined = random_local_attn + gate * memory`. Память должна компенсировать этот шум.

**Решение:** В `FullParamsInit` (ноутбук) скопировать веса `attn` → `local_attn` после загрузки Gemma. Тогда замороженный `local_attn` производит осмысленный near-context, а не шум.

---

### Предлагаемые изменения

#### 1. `routing_optimizer.py` — критический фикс

```python
mask=kd.optim.select([
    "memory", "memory_gate_proj", "local_attn",          # ← ДОБАВЛЕНО
    "titans_ffn", "titans_pre_ffw_norm", "titans_post_ffw_norm",
    "layer_24", "layer_25", "final_norm"
]),
```

`local_attn` попадёт в `adam_base_mask` (Q/K/V/O проекции не в `M3_KEYS`), будет обучаться через Adam-atan2 с `lr_adam`. Это разумно — local_attn ближе к стандартному attention, чем к NeuralMemory проекциям.

#### 2. Phase 1 `experimental_config` — без изменений

```python
experimental_config = {
    'heads': 8,
    'dim_head': 128,
    'chunk_size': 32,
    'mlp_depth': 6,
    'max_grad_norm': 0.5,
    'elastic_net_lambda': 0.001,
    'diff_view': False,
    'is_look_ahead': False,
    'huber_loss_delta': None,
    'adaptive_max_lr': 1e-4,  # без изменений
}
```
`local_attn` заморожен → `neural_mem_kwargs` не меняются. Единственное изменение — инициализация в FullParamsInit (см. ниже).

#### 3. Phase 1 `opt_params` — без изменений

`local_attn` заморожен через `stop_gradient`, не нуждается в optimizer params.

#### 4. Phase 1 `FullParamsInit` — инициализация из Gemma

Добавить логику копирования `attn` → `local_attn`:
```python
# В FullParamsInit.transform(), после merged = _deep_merge(state.params, self.params):
for key, val in merged.items():
    if isinstance(val, Mapping) and 'local_attn' in val and 'attn' in val:
        # Копируем веса глобального attention в локальный
        val['local_attn'] = copy.deepcopy(val['attn'])
```

#### 5. Phase 2 `experimental_config` — мягкая коррекция

```python
experimental_config = {
    'heads': 8,
    'dim_head': 128,
    'chunk_size': 32,
    'mlp_depth': 6,
    'max_grad_norm': 0.5,
    'elastic_net_lambda': 0.0,
    'huber_loss_delta': 0.1,
    'diff_view': False,
    'is_look_ahead': False,
    'adaptive_max_lr': 3e-4,   # БЫЛО 5e-4 → СНИЖЕНО: local_attn берёт часть работы на себя
}
```

**Обоснование снижения `adaptive_max_lr`:** Теперь local_attn обеспечивает near-context, и памяти не нужно работать так агрессивно. Снижение с 5e-4 до 3e-4 уменьшит drift памяти между шагами оптимизатора, что особенно важно при `every_k_schedule=4`.

#### 6. Phase 2 `opt_params` — ключевое изменение

```python
opt_params = {
    "lr_muon": 1e-5,              # без изменений (M3 для NeuralMemory проекций)
    "beta": 0.90,
    "lr_adam": 2e-4,              # БЫЛО 5e-4 → СНИЖЕНО: теперь применяется и к local_attn
                                 # local_attn стартует из предобученных весов Gemma — нужен мягкий LR
    "adam_b1": b1_schedule,
    "adam_b2": 0.85,
    "lr_gate": 2e-3,              # без изменений
    "gate_b1": b1_schedule,
    "gate_b2": 0.95,
    "every_k_schedule": 4,
}
```

**Обоснование снижения `lr_adam`:** `lr_adam` теперь управляет не только memory params, но и `local_attn` (Q/K/V/O проекции). Поскольку local_attn инициализируется из Gemma, агрессивный LR разрушит предобученные признаки. 2e-4 — компромисс между адаптацией и сохранением.

**Альтернатива (более точная, но сложная):** Создать отдельный optimizer branch для `local_attn` с собственным `lr_attn = 1e-4`. Это требует изменения `routing_optimizer.py` — добавить 4-ю маску.

---

### Оценка потребления памяти TPU

#### Параметры local_attn (Gemma3-1B, embed_dim=1152, 4 heads, head_dim=256)

| Компонент | Форма | Параметров | bf16 |
|-----------|-------|-----------|------|
| q_e (kernel) | (1152, 1024) | 1.18M | 2.4 MB |
| k_e (kernel) | (1152, 256) | 295K | 0.6 MB |
| v_e (kernel) | (1152, 256) | 295K | 0.6 MB |
| attn_output (kernel) | (1024, 1152) | 1.18M | 2.4 MB |
| **Итого на слой** | | **~2.95M** | **~6.0 MB** |

#### Phase 1 (1 слой, local_attn заморожен)

| Ресурс | Без local_attn | С local_attn | Δ |
|--------|---------------|-------------|---|
| Параметры модели | ~500M | ~503M | +0.6% |
| Optimizer state | — | — (заморожен) | 0 |
| KV cache (доп.) | — | +10.5 MB | ~0.3% от HBM |
| Activations (forward) | — | +52 MB | ~0.2% от HBM |
| **TPU HBM total** | ~baseline | **+~0.5%** | пренебрежимо |
| **Compile RAM (XLA)** | ~5 GB | **~6.5-7 GB** | **+30-40%** |

Увеличение compile RAM связано с дополнительным XLA-графом для sliding window attention (matmuls + softmax + masking). С `flax_nn.remat` backward рекомпутит эти операции.

#### Phase 2 (3 слоя: 11, 17, 23 — local_attn обучаемый)

| Ресурс | Без local_attn | С local_attn | Δ |
|--------|---------------|-------------|---|
| Параметры (3 слоя) | baseline | +8.85M | +1.8% |
| Optimizer state (Adam, 2× params, float32) | — | +70.8 MB | ~0.2% от HBM |
| KV cache (3 слоя) | — | +31.5 MB | ~0.1% от HBM |
| Activations (forward, remat) | — | +156 MB (peak) | ~0.5% от HBM |
| **TPU HBM total** | ~baseline | **+~1%** | пренебрежимо |
| **Compile RAM (XLA)** | ~70 GB (3 слоя) | **~91-98 GB** | **+30-40%** |

**Ключевой риск:** Compile RAM (CPU RAM) при `titans_first_layer=11` может достичь ~95 GB. Colab TPU обычно имеет ~100-200 GB CPU RAM. Возможен OOM при компиляции.

**Митигация:** 
- Начать с `titans_first_layer=17` (2 слоя): compile RAM ~33-35 GB
- Или `titans_first_layer=23` (1 слой): compile RAM ~7 GB — безопасно

#### HBM (TPU v6e, 32 GB) —不用担心

local_attn добавляет <1% к HBM. Основные потребители HBM (KV-cache для всех 26 слоёв, NeuralMemory state) остаются неизменными. Sliding window attention (window=128) в 8× легче global attention по вычислениям активаций.

---

### Сводка изменений для применения

| Файл | Изменение | Приоритет |
|------|-----------|-----------|
| `routing_optimizer.py` | Добавить `"local_attn"` в `partial_updates` select | 🔴 Критический |
| Phase 1 ноутбук | Копировать `attn`→`local_attn` в `FullParamsInit` | 🟡 Важно |
| Phase 2 ноутбук | `adaptive_max_lr`: 5e-4 → 3e-4 | 🟢 Рекомендуется |
| Phase 2 ноутбук | `lr_adam`: 5e-4 → 2e-4 | 🟢 Рекомендуется |
| Phase 2 ноутбук | Начать с `titans_first_layer=23` | 🟢 Безопасность |

---

Если план одобрен, переключите в **Act mode**, и я внесу изменения в `routing_optimizer.py` и подготовлю фрагменты кода для ноутбуков.