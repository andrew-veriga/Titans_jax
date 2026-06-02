# M3 Optimizer: Multi-Scale Momentum Muon

> **Статус:** Используется в Phase 1training через `routing_optimizer.py`
> **Источник теории:** "Nested Learning: The Illusion of Deep Learning Architectures" (Behrouz et al., NeurIPS 2025)

---

## 1. Мотивация

Что находится **внутри NeuralMemory** и как это обновляется:

## Два разных типа «весов» в TitansBlock

### Тип 1: Внутренние веса памяти (MLP weights) — `memory_state`
Это `weight_0`, `weight_1` матрицы формы `(B, H, dim_head, dim_head)`. Они:
- **Обновляются per-example** через `associative_scan` в `store_memories()`
- **Сбрасываются** между примерами (cache=None в TrainingOutput → `init_state()` вызывается заново)
- **НЕ видны** внешнему оптимизатору вообще — они не часть `model.params`

### Тип 2: Проекционные слои (внешние параметры) — `model.params`
Это Dense-слои, которые определяют **как** кодировать/декодировать информацию:
- `to_queries` → проекторы для извлечения из памяти
- `to_keys_values` → проекторы для записи в память
- `combine_heads` → объединение голов
- `to_momentum`, `to_adaptive_step`, `to_decay_factor` → контролируют dynamics памяти
- `memory_gate_proj` → вентиль (Adam-atan2)
- norms, pooling layers и т.д.

**Эти** параметры оптимизируются **между примерами** через внешний оптимизатор.

Функционал MemoryMLP
MemoryMLP — это «ядро» нейронной памяти, маленькая нейросеть внутри памяти, которая преобразует запросы (queries) в извлечённые значения. Её веса — это те самые weight₀ и weight₁ матрицы формы (H, dim_head, dim_head).

Архитектура

class MemoryMLP(nn.Module):
    dim_head: int
    depth: int = 2  # mlp_depth
    
    def __call__(self, x):
        for i in range(self.depth):
            w = self.param(f'weight_{i}', ...)
            x = x @ w                    # линейная проекция
            if i < self.depth - 1:
                x = jax.nn.silu(x)       # активация между слоями
        return x
При depth=2: q → (q @ W₀ + silu) → (· @ W₁) → retrieved_value

Два режима работы
1. Retrieve (чтение):


retrieved = memory_model.apply({'params': W}, query)
Query-вектор умножается на матрицы W₀, W₁ — аналог «чтения из памяти по адресу query».

2. Store (запись) через gradient descent:


W_new = W - lr * ∇W loss(W, key, value)
Матрицы W₀/W₁ обновляются внутри associative_scan через градиентный шаг по лоссу (MSE или Huber) между предсказанием и целевым значением. Это эквивалентно «записи key→value в память».

Ключевое свойство
MemoryMLP — это parameterized associative memory: она ведёт себя как словарь (key→value), но реализована как нейросеть. В отличие от стандартного Key-Value Attention:

KV-Attention	MemoryMLP
Хранит пары (K,V) явно	Хранит знания в весах W₀,W₁
Exact match lookup	Approximate pattern-based lookup
Ёмкость = размер KV-кэша	Ёмкость = O(dim²) параметров
Не обобщает	Обобщает паттерны через обучение
По сути, MemoryMLP — это «сжатая» память: вместо хранения всех токенов напрямую, она кодирует паттерны в весах матриц через online gradient descent.

Веса MemoryMLP инициализируются __один раз__ как параметры модели (через `nn.param`), а затем __копируются__ для каждого примера в `memory_state`.

Вот как это работает:

### 1. Начальная инициализация (при создании модели)

```python
class MemoryMLP(nn.Module):
    def __call__(self, x):
        for i in range(self.depth):
            w = self.param(f'weight_{i}', nn.initializers.lecun_normal(), ...)
```

`nn.param` создаёт W₀, W₁ __один раз__ при инициализации модели — через LeCun normal. Они становятся частью `model.params` (дерева параметров Flax).

### 2. Копирование в memory_state для каждого батча

В `init_memory_state()` (вызывается когда `cache=None`):

```python
def init_memory_state(batch_size, dim, neural_mem_kwargs, dtype):
    ...
    # memory_model.params → копируется во внутрение веса памяти
    memory_state = {
        'weight_0': jnp.tile(model.params['weight_0'], (batch_size, 1, 1, 1)),
        'weight_1': jnp.tile(model.params['weight_1'], (batch_size, 1, 1, 1)),
    }
```

Форма меняется от `(H, d, d)` → `(B, H, d, d)`: каждый пример в батче получает __свою копию__ весов.

### 3. Обновление внутри примера

В течение одного примера (последовательность токенов) веса обновляются через `associative_scan`:

```javascript
W₀ ← W₀ - lr * ∇loss(W₀, key_chunk, value_chunk)  # для каждого чанка
```

### 4. Сброс между примерами

Когда `cache=None` (новый пример) → `init_memory_state()` вызывается снова → веса __снова копируются из `model.params`__ (исходных, не изменённых).

### Итого lifecycle:

``` Python
model.params (обучаемые внешним оптимизатором)
    │
    ├─→ init_memory_state() → копия (B,H,d,d)  ← для примера #1
    │       ↓
    │   associative_scan: W обновляется per-chunk
    │       ↓
    │   пример #1 завершён → memory_state отбрасывается
    │
    ├─→ init_memory_state() → свежая копия      ← для примера #2
    │       ↓
    │   associative_scan: W обновляется per-chunk
    │       ↓
    │   пример #2 завершён → memory_state отбрасывается
    │
    └─→ ... и так далее

Внешний оптимизатор (M3/Adam) обновляет model.params между шагами
→ следующий init_memory_state() берёт обновлённые веса
```

__Суть:__ MemoryMLP — это обучаемый «шаблон» памяти. Его веса оптимизируются внешним оптимизатором (не случайно!), а для каждого примера создаётся временная копия, которая «адаптируется» к конкретной последовательности

``` mermaid
flowchart TB
    subgraph TitansBlock
        direction TB
        Input["x_input"]
        subgraph Memory_Codec["Кодек памяти (M3)"]
            Q["to_queries"]
            KV["to_keys_values"]
            CH["combine_heads"]
        end
        subgraph Memory_Controllers["Контроллеры (Adam-atan2)"]
	        direction TB
            MOM["to_momentum"]
            ADAPT["to_adaptive_step"]
            DECAY["to_decay_factor"]
        end
        Gate["memory_gate_proj<br/>(Adam-atan2)"]
        subgraph NeuralMemory["Нейронная память (NLM)"]
            Store["store_memories()"]
            W0["MemoryMLP: weight₀ (B,H,d,d)"]
            W1["MemoryMLP: weight₁ (B,H,d,d)"]
            Retrieve["retrieve_memories() query × W"]
        end
        Add1["+residual"]
        FFN["FFN (стандартный Transformer)"]
        Add2["+residual"]
        Output["x_output"]
    end
    Input --> Q & KV & Gate
    Q --> Retrieve
    KV --> Store
    MOM & ADAPT & DECAY --> Store
    Store --> W0 & W1
    W0 & W1 --> Retrieve
    Retrieve --> CH --> Gate --> Add1
    Input --> Add1
    Add1 --> FFN --> Add2
    Add1 --> Add2 --> Output
```
### Маршрутизация оптимизатора:

| Параметры                                                                            | Оптимизатор                 | Что делает                  |
| ------------------------------------------------------------------------------------ | --------------------------- | --------------------------- |
| `to_queries`, `to_keys_values`, `combine_heads`                                      | **M3** (slow+fast momentum) | Проекции «кодека» памяти    |
| `memory_gate_proj`                                                                   | **Adam-atan2**              | Вентиль gate                |
| Всё остальное (`to_momentum`, `to_adaptive_step`, `to_decay_factor`, norms, pooling) | **Adam-atan2**              | Контроллеры динамики памяти |

### Итого: что обучается медленным моментумом?

**M3 с `slow_update_freq=8`** обучает только **3 проекционных ядра**:
1. `to_queries/kernel` — как формировать запросы к памяти
2. `to_keys_values/kernel` — как формировать ключи/значения для записи
3. `combine_heads/kernel` — как объединять выходы голов

Это «кодек» памяти — он определяет **как** информация преобразуется перед попаданием в память и после извлечения. Медленный моментум здесь полезен: проекции должны быть стабильными across many diverse examples, а не подстраиваться под каждый отдельный пример.

### Проблема: катастрофическое забывание оптимизатора

Стандартный Adam/SGD с моментумом эквивалентен **контекстному окну размером ~40 шагов** (при β=0.9). При обучении на последовательных данных с разными задачами/чанками:

1. Оптимизатор «забывает» ландшафт предыдущих задач
2. Переключение на новую задачу → потеря информации о старом направлении
3. Это **катастрофическое забывание оптимизатора** (не модели!)

### Решение: Continuum Memory System (CMS)

CMS — спектр модулей с разными частотами обновления:
- **Быстрый модуль** (M¹): адаптируется к локальному ландшафту, обновляется каждый шаг
- **Медленный модуль** (M²): хранит сжатую сводку ландшафта, обновляется каждые Ĉ шагов
- **Агрегация:** Newton-Schulz + взвешенная сумма

---

## 2. Архитектура M3

### Формула агрегации

```
m_combined = newton_schulz(m_fast, m_slow) * (1 - slow_weight) + m_slow * slow_weight
```

Где:
- `m_fast` — стандартный моментум Adam (обновляется каждый шаг)
- `m_slow` — медленный моментум (обновляется каждые `slow_update_freq` шагов)
- `newton_schulz` — 5 итераций, ортогонализация матрицы моментума
- `slow_weight` — вес медленного момента в итоговой агрегации

### Newton-Schulz итерация

```python
def newton_schulz(M, steps=5):
    """Ортогонализация моментума через Newton-Schulz итерации."""
    a = (3 - b * c) / 2  # где b = M^T M, c = M M^T
    # После 5 итераций: M → ближайшая ортогональная матрица
```

Это аналог Muon optimizer из статьи Nested Learning (§3.4, приложение 5).

---

## 3. Реализация в Titans_jax

### Файлы

| Файл | Назначение |
|------|-----------|
| `m3_optimizer.py` | Ядро M3: агрегация fast/slow моментума |
| `routing_optimizer.py` | Маршрутизация оптимизаторов по типам параметров |

### Три ветки оптимизатора

В Phase 1 training параметры модели разделены на 3 группы, каждая со своим оптимизатором:

```
┌──────────────────────────────┬───────────────────────┬───────────────────────────┐
│ Группа параметров            │ Оптимизатор           │ Ключи                     │
├──────────────────────────────┼───────────────────────┼───────────────────────────┤
│ Memory Codec                 │  M3 (Muon + CMS)      │ *to_queries*              │
│                              │                       │ *to_keys_values*          │
│                              │                       │ *combine_heads*           │
├──────────────────────────────┼───────────────────────┼───────────────────────────┤
│ Gate                         │ Adam-atan2            │ *gate*                    │
├──────────────────────────────┼───────────────────────┼───────────────────────────┤
│ Memory Controllers           │ Adam-atan2            │ *to_momentum*             │
│                              │                       │ *to_adaptive_step*        │
│                              │                       │ *to_decay_factor*         │
│                              │                       │ *chunk_pool*              │
│                              │                       │ norms, pooling и т.д.     │
├──────────────────────────────┼───────────────────────┼───────────────────────────┤
│ Остальные (Gemma frozen)     │ Adam-atan2 (freeze)   │ всё остальное             │
└──────────────────────────────┴───────────────────────┴───────────────────────────┘
```

### Параметры M3

| Параметр | Default | Описание |
|----------|---------|----------|
| `lr_muon` | 5e-4 → 1e-5 (cosine) | Learning rate для M3 ветки |
| `beta` | 0.90 | β₁ для fast моментума |
| `slow_update_freq` | 8 | Частота обновления slow моментума |
| `slow_weight` | 0.1 | Вес slow моментума в агрегации |
| `every_k_schedule` | 4 | Планировщик частоты обновления памяти |

### Параметры Gate (Adam-atan2)

| Параметр | Default | Описание |
|----------|---------|----------|
| `lr_gate` | 5e-3 → 5e-4 (cosine) | Learning rate для gate |
| `gate_b1` | schedule 0.7→0.9 | β₁ для gate |
| `gate_b2` | 0.95 | β₂ для gate |

### Параметры Adam-atan2 (base/freeze)

| Параметр | Default | Описание |
|----------|---------|----------|
| `lr_adam` | 1e-4 → 1e-5 (cosine) | Learning rate для Adam |
| `adam_b1` | schedule 0.7→0.9 | β₁ |
| `adam_b2` | 0.85 | β₂ |

---

## 4. Практическое влияние `slow_weight`

### slow_weight = 0.0 (без CMS)

```
m_combined = newton_schulz(m_fast) * 1.0
```

Чистый Muon без медленного моментума. Оптимизатор «видит» только последние ~40 шагов.

### slow_weight = 0.1 (default)

```
m_combined = newton_schulz(m_fast, m_slow) * 0.9 + m_slow * 0.1
```

10% вклада от медленного моментума. Начинает учитываться долгосрочный ландшафт.

### slow_weight = 0.3 (агрессивный CMS)

```
m_combined = newton_schulz(m_fast, m_slow) * 0.7 + m_slow * 0.3
```

30% вклада от медленного моментума. Значительно лучшее сохранение информации о предыдущих задачах/чанках.

### Рекомендация

Для Phase 1 (послойная дистилляция на OpenWebText):
- Начинать с `slow_weight = 0.1`, `slow_update_freq = 16`
- Если loss нестабилен или «пила» на графике → увеличить до `0.2-0.3`
- Уменьшить `slow_update_freq` до `4-8` для более частого обновления медленного момента

---

## 5. Связь с теорией

### Из статьи "Nested Learning" (§3.4)

M3 реализует расширения моментума:

| Расширение | Реализация в M3 |
|-----------|-----------------|
| More Expressive Association | Matrix-valued momentum (не scalar) |
| More Expressive Objectives | Delta Momentum (content-dependent decay) |
| More Expressive Memory | CMS с двумя масштабами |
| Higher-order Feature Maps | Newton-Schulz ортогонализация |
| Nonlinear Outputs | Newton-Schulz как нелинейная агрегация |

### Ключевое свойство

> "Architecture generates context for the optimizer" — модель и оптимизатор образуют взаимосвязанную систему. M3 понимает структуру модели (memory параметры vs attention параметры) и адаптирует стратегию оптимизации.

---

## 6. Интеграция с training loop

### Использование в ноутбуке Phase 1

```python
from routing_optimizer import make_routing_optimizer

opt_params = {
    "lr_muon": optax.warmup_cosine_decay_schedule(...),
    "beta": 0.90,
    "lr_adam": optax.warmup_cosine_decay_schedule(...),
    "adam_b1": b1_schedule,
    "adam_b2": 0.85,
    "lr_gate": optax.warmup_cosine_decay_schedule(...),
    "gate_b1": b1_schedule,
    "gate_b2": 0.95,
    "slow_update_freq": 8,
    "slow_weight": 0.1,
    "every_k_schedule": 4,
}

routing_optimizer = make_routing_optimizer(opt_params)
```

### Параметры, которые НЕ входят в M3

Следующие параметры управляют нейронной памятью (архитектурой), а не оптимизатором:

```python
experimental_config = {
    'heads': 8,
    'dim_head': 128,
    'chunk_size': 32,
    'mlp_depth': 2,
    'max_grad_norm': 0.5,
    'elastic_net_lambda': 0.01,
    'adaptive_max_lr': 1e-4,
}
```

---

## 7. См. также

- [training_phases.md](training_phases.md) — общее описание фаз обучения
- [NeuralMemory_Parameters_Training.md](NeuralMemory_Parameters_Training.md) — параметры нейронной памяти
- [ titans-wiki: Nested Learning и Hope](concepts/Nested Learning and Hope.md) — теоретическая основа M3/CMS