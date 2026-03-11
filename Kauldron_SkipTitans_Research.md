# Исследование: Реализация `SkipTitans` InitTransform для Kauldron / Gemma

## Контекст задачи
В экосистеме DeepMind (Kauldron + JAX) для загрузки предобученных моделей с адаптерами (например, LoRA) используются механизмы `InitTransform`. Пользователь привел пример из официальных туториалов:

```python
init_transform=gm.ckpts.SkipLoRA(
    wrapped=gm.ckpts.LoadCheckpoint(
        path=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
    )
)
```

**Цель:** Выяснить, можно ли написать аналогичный класс `SkipTitans` для загрузки официальных весов Gemma, пропуская и оставляя случайно инициализированными новые кастомные модули Neural Memory (`memory` и `memory_gate`), чтобы использовать это в Kauldron Trainer.

## Анализ архитектуры `SkipLoRA`
В ходе исследования исходного кода библиотеки `gemma` (`venv_312/Lib/site-packages/gemma/gm/ckpts/_lora.py` и `gemma/peft/_tree_utils.py`) было установлено следующее:

1. **Базовый класс:** `SkipLoRA` наследуется от интерфейса `kd.ckpts.AbstractPartialLoader` из библиотеки Kauldron.
2. **Алгоритм работы метода `transform`:**
   - Принимает текущее состояние обучения `state` (где модель уже имеет инициализированные случайные параметры, включая новые модули LoRA).
   - Вызывает утилиту `peft.split_params(state.params)`, которая рекурсивно обходит PyTree (словарь параметров) и разделяет его на два дерева: `original_params` (базовые веса) и `lora_params` (добавленные веса по ключу `lora`).
   - Подменяет параметры в стейте: `state = state.replace(params=original_params)`.
   - Делегирует загрузку "обернутому" лоадеру (обычно `LoadCheckpoint`): `state = self.wrapped.transform(state)`. Загрузчик корректно отрабатывает, так как ожидаемые в чекпоинте ключи теперь совпадают со структурой `state.params`.
   - Выполняет обратное слияние (`peft.merge_params`), "приклеивая" случайно инициализированные `lora_params` к загруженным весам чекпоинта: `state = state.replace(params=merged_params)`.

## Прототип реализации `SkipTitans`

Паттерн `SkipLoRA` идеально подходит для нашей задачи. Мы можем реализовать собственную логику фильтрации дерева параметров (PyTree), заменяя поиск ключа `lora` на ключи `memory` и `memory_gate`.

### 1. Утилиты разделения и слияния дерева (PyTree)
Для начала необходимо реализовать рекурсивный обход словаря параметров для изоляции модулей Titans:

```python
def split_titans_params(params):
    """
    Разделяет дерево параметров на базовую Gemma и новые модули Titans.
    """
    original_tree = {}
    titans_tree = {}

    def _split_recursive(input_subtree, original_subtree, titans_subtree):
        for key, value in input_subtree.items():
            if isinstance(value, dict):
                # Если узел - это модуль памяти или гейт
                if 'memory' in key or 'memory_gate' in key:
                    titans_subtree[key] = value
                else:
                    original_subtree[key] = {}
                    titans_subtree[key] = {}
                    _split_recursive(value, original_subtree[key], titans_subtree[key])
            elif 'memory' in key or 'memory_gate' in key:
                titans_subtree[key] = value
            else:
                original_subtree[key] = value

    _split_recursive(params, original_tree, titans_tree)
    
    # Очистка пустых словарей из titans_tree
    def _remove_empty_dicts(tree):
        if not isinstance(tree, dict):
            return tree
        new_tree = {}
        for key, value in tree.items():
            if isinstance(value, dict):
                sub_tree = _remove_empty_dicts(value)
                if sub_tree:
                    new_tree[key] = sub_tree
            else:
                new_tree[key] = value
        return new_tree

    titans_tree = _remove_empty_dicts(titans_tree)
    return original_tree, titans_tree

def merge_titans_params(original_params, titans_params):
    """
    Выполняет слияние базовых весов и весов Titans.
    (Можно реализовать аналогично функции stitch_hybrid_model из MVP)
    """
    def _merge_recursive(orig_subtree, tit_subtree):
        new_tree = {}
        for key, value in orig_subtree.items():
            if isinstance(value, dict) and key in tit_subtree:
                new_tree[key] = _merge_recursive(value, tit_subtree[key])
            else:
                new_tree[key] = value
        for k in sorted(set(tit_subtree) - set(orig_subtree)):
            new_tree[k] = tit_subtree[k]
        return new_tree

    return _merge_recursive(original_params, titans_params)
```

### 2. Реализация класса InitTransform

Имея функции разделения, создание трансформера для Kauldron выглядит так:

```python
import dataclasses
from kauldron import kd

@dataclasses.dataclass(frozen=True)
class SkipTitans(kd.ckpts.AbstractPartialLoader):
    """Обертка над PartialLoader для сохранения случайной инициализации блоков Titans."""
    wrapped: kd.ckpts.AbstractPartialLoader

    def transform(self, state: kd.train.TrainState) -> kd.train.TrainState:
        # 1. Отделяем случайно инициализированные веса памяти
        original_params, titans_params = split_titans_params(state.params)
        
        # 2. Подменяем стейт на чистую архитектуру Gemma
        state = state.replace(params=original_params)
        
        # 3. Загружаем базовую модель через LoadCheckpoint (wrapped)
        state = self.wrapped.transform(state)
        
        # 4. Склеиваем загруженную базу со случайными весами памяти
        merged_params = merge_titans_params(state.params, titans_params)
        
        return state.replace(params=merged_params)
```

### 3. Пример предполагаемого использования
При переходе на Kauldron Trainer конфигурация выглядела бы декларативно:

```python
init_transform = SkipTitans(
    wrapped=gm.ckpts.LoadCheckpoint(
        path=CKPT_PATH,
    )
)

trainer = kd.train.Trainer(
    model=model,
    init_transform=init_transform,
    # ... остальные параметры Kauldron
)
```

## Вывод
Реализация механизма "в стиле DeepMind" для нестандартных архитектур в JAX абсолютно жизнеспособна. Паттерн `SkipLoRA` легко адаптируется под любую кастомную структуру весов (`SkipTitans`). В рамках MVP проекта предпочтение было отдано "ручной" сшивке PyTree (на уровне словарей Optax), однако данный механизм может быть использован на этапе масштабирования проекта и перехода на использование библиотеки Kauldron.