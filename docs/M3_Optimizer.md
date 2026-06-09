# M3 Optimizer: Multi-Scale Momentum Muon

> **Статус:** Используется в Phase 1training через `routing_optimizer.py`
> **Источник теории:** "Nested Learning: The Illusion of Deep Learning Architectures" (Behrouz et al., NeurIPS 2025)

---

## 1. Мотивация


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