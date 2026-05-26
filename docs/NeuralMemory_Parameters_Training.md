# Где обучаются и сохраняются параметры NeuralMemory при маске `('memory', 'memory_gateproj')`

## Краткий ответ: ВСЕ эти параметры **обучаются и сохраняются** ✅

Они находятся внутри Flax-поддерева `memory.params.*`, которое попадает под маску `"memory"` в `kd.optim.select(["memory", "memory_gateproj"])`.

---

## Как работает маска оптимизатора (`routing_optimizer.py`, строки 154–159)

```python
optax.MultiSteps(
    kd.optim.partial_updates(
        inner_chain,
        mask=kd.optim.select(["memory", "memory_gateproj"]),
    ),
    every_k_schedule=...,
)
```

`partial_updates` (`_freeze.py`) превращает это в `optax.multi_transform`:
- **`'train'`** → `inner_chain` (реальный оптимизатор) — для параметров, соответствующих паттернам `"memory"` или `"memory_gate_proj"`
- **`'freeze'`** → `optax.set_to_zero()` — для всех остальных (нулевой апдейт = заморожены)

`select(["memory", "memory_gate_proj"])` (`_masks.py`) создаёт regex-паттерны:
- `"memory"` → `r"(?:^|\.)memory(?:$|\.)"` — совпадает с любым путём, содержащим `.memory.`
- `"memory_gate_proj"` → `r"(?:^|\.)memory\_gate\_proj(?:$|\.)"` — аналогично

---

## Дерево параметров TitansBlock (пример для `layer_23`)

```
params['layer_23'] = {
    'pre_attention_norm': { 'scale': ... },     # ❌ FROZEN
    'post_attention_norm': { 'scale': ... },     # ❌ FROZEN
    'pre_ffw_norm': { 'scale': ... },            # ❌ FROZEN
    'post_ffw_norm': { 'scale': ... },            # ❌ FROZEN
    'mlp': { ... },                               # ❌ FROZEN (Gemma MLP)
    'memory': {                                    # ✅ TRAINED (matched "memory")
        'params': {
            'to_queries':       { 'kernel': ... },  # ✅ M3 optimizer
            'to_keys':          { 'kernel': ... },  # ✅ Adam-atan2 (base) *
            'to_keys_values':   { 'kernel': ... },  # ✅ M3 optimizer
            'to_momentum':      { 'kernel': ... },  # ✅ Adam-atan2 (base)
            'to_adaptive_step': { 'kernel': ... },  # ✅ Adam-atan2 (base)
            'to_decay_factor':  { 'kernel': ... },  # ✅ Adam-atan2 (base)
            'combine_heads':    { 'kernel': ... },  # ✅ M3 optimizer
            'retrieve_norm':    { 'scale': ... },   # ✅ Adam-atan2 (base)
            'store_norm':       { 'scale': ... },   # ✅ Adam-atan2 (base)
            'multihead_rmsnorm':{ 'gamma': ... },   # ✅ Adam-atan2 (base)
            'memory_model':     { 'weight_0': ..., 'weight_1': ... },  # ✅ Adam-atan2 (base)
            'empty_memory_embed': ...,               # ✅ Adam-atan2 (base)
            'chunk_pool_layer1': { 'kernel': ... },  # ✅ Adam-atan2 (base)
            'chunk_pool_layer2': { 'kernel': ... },  # ✅ Adam-atan2 (base)
            'retrieve_gate':    { ... },             # ✅ Adam-atan2 (base) (если heads > 1)
        }
    },
    'memory_gate_proj': {                          # ✅ TRAINED (matched "memory_gate_proj")
        'kernel': ...,                              # ✅ Adam-atan2 (gate)
        'bias': ...,                                # ✅ Adam-atan2 (gate)
    },
}
```

## Маршрутизация внутри обучаемых параметров (3-way routing)

Внутри `inner_chain` параметры распределяются по 3 оптимизаторам (`routing_optimizer.py`, строки 31–65):

| Параметр | M3 (Muon-подобный) | Adam-atan2 (gate) | Adam-atan2 (base) |
|---|---|---|---|
| `to_queries.kernel` | ✅ | | |
| `to_keys_values.kernel` | ✅ | | |
| `combine_heads.kernel` | ✅ | | |
| `memory_gate_proj.kernel/bias` | | ✅ | |
| `to_keys.kernel` | | | ✅ |
| `to_momentum.kernel` | | | ✅ |
| `to_adaptive_step.kernel` | | | ✅ |
| `to_decay_factor.kernel` | | | ✅ |
| всё остальное в `memory.*` | | | ✅ |

## ⚠️ Замечание о `to_keys`

Слой `self.to_keys` (строка 307 в `titans.py`) **объявлен**, но **нигде не используется** — `store_memories` использует только `self.to_keys_values`, а `retrieve_memories` — только `self.to_queries`. Его параметр `kernel` всё равно существует в дереве параметров и обучается через Adam-atan2 (base), но не влияет на forward pass. Это можно считать мёртвым параметром.

## Сохранение в чекпоинты

Все обучаемые параметры (включая все `to_*` слои) сохраняются в чекпоинты Kauldron вместе с состоянием оптимизатора, так как входят в `params` и `opt_state` модели.

---

## Источники

- `titans.py` — класс `NeuralMemory`, объявление всех `to_*` слоёв (строки 306–313)
- `gemma_titans.py` — класс `TitansBlock`, объявление `memory` и `memory_gate_proj` (строки 105–123)
- `routing_optimizer.py` — маски `m3_mask`, `gate_mask`, `adam_base_mask` и функция `make_routing_optimizer` (строки 31–159)
- `kauldron/optim/_masks.py` — функция `select()` с regex-матчингом путей
- `kauldron/optim/_freeze.py` — функция `partial_updates()` → `optax.multi_transform`