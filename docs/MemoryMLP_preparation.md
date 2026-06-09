Проект Titans. Что находится **внутри NeuralMemory**  и как она обновляется:

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