# Migration Guide: Phase 3 — Precomputed Activations Training

## Обзор

**Проблема:** XLA-граф Gemma 1B с backward через 22 замороженных слоя не помещается в HBM TPU v6.

**Решение:** Предвычислить вход слоя 23 офлайн, обучать только TitansBlock + head.

```
БЫЛО (Фаза 2):
  tokens → [22 замороженных слоя] → [TitansBlock] → [3 слоя + head] → loss
           ↑ XLA компилирует ВСЁ, OOM

СТАЛО (Фаза 3):
  Шаг 1 (офлайн): tokens → [22 слоя] → hidden_state → сохранить на диск
  Шаг 2 (обучение): hidden_state → [TitansBlock] → [3 слоя + head] → loss
                     ↑ XLA компилирует ТОЛЬКО 4 слоя ✅
```

## Быстрый старт

### 1. Предвычислить активации

```bash
python precompute_activations.py \
    --gemma_ckpt /path/to/gemma-3-1b-pt \
    --dataset_repo veriga/openwebtext-gemma3-tokenized-1024 \
    --output_dir ./activations_layer23 \
    --target_layer 23 \
    --batch_size 8 \
    --max_seq_len 1024
```

Результат: `./activations_layer23/shard_*.npy` (~36GB для 8M примеров)

### 2. Обучить TitansBlock

```bash
python train_layer23.py \
    --gemma_ckpt /path/to/gemma-3-1b-pt \
    --activation_dir ./activations_layer23 \
    --token_dataset_repo veriga/openwebtext-gemma3-tokenized-1024 \
    --output_dir ./checkpoints_layer23 \
    --mlp_depth 6 \
    --batch_size 4 \
    --num_steps 10000
```

### 3. Проверить сэмплинг

После обучения — собрать гибридную модель и запустить генерацию.

## Детали архитектуры

### Что обучается, что заморожено

```
Слои 0-22:   Gemma attention (ЗАМОРОЖЕНЫ, предвычислены)
Слой 23:     TitansBlock (ОБУЧАЕТСЯ)
  ├── NeuralMemory (mlp_depth=6, heads=8, dim_head=256)
  ├── memory_gate_proj (Dense, embed_dim)
  ├── pre_attention_norm (RMSNorm)
  ├── MLP (FeedForward)
  └── post norms
Слои 24-25:  Gemma attention (ЗАМОРОЖЕНЫ, в графе обучения)
Final norm:  RMSNorm (ЗАМОРОЖЕН)
Head:        embedder.decode (ЗАМОРОЖЕН)
```

### Формат данных

| Файл | Формат | Размер |
|------|--------|--------|
| `shard_XXXXXX.npy` | `(batch_per_shard, 1024, 1152)` float32 | ~36 MB/shard (bs=8) |
| `metadata.json` | конфигурация + прогресс | ~1 KB |

### Параметры TitansBlock (рекомендуемые)

```python
neural_mem_kwargs = {
    'heads': 8,
    'dim_head': 256,
    'chunk_size': 32,
    'mlp_depth': 6,         # ↑ с 2 до 6 — больше ёмкость для семантики
    'max_grad_norm': 0.5,
    'adaptive_max_lr': 1e-3,
    'every_k_schedule': 1,  # без MultiSteps — граф маленький
}
```

## Ключевые отличия от Фазы 1/2

| Аспект | Фаза 1 | Фаза 2 | Фаза 3 (новая) |
|--------|--------|--------|----------------|
| Слои Titans | 4 (5,11,17,23) | 4 (5,11,17,23) | **1 (только 23)** |
| Обучаемые параметры | Проекции NLM | Titans + gate | **Titans + gate** |
| Teacher | Global attention | Нет | **Нет** |
| Loss | MSE скрытых состояний | CrossEntropy | **CrossEntropy** |
| Вход | Токены → вся модель | Токены → вся модель | **Предвычисленные векторы** |
| XLA граф | 1 слой (параллельно) | 23+ слоёв (OOM!) | **4 слоя (помещается!)** |
| NLM depth | 2 | 2 | **6** |

## Следующие шаги (после success)

1. **Валидация:** Сэмплинг связного текста → proof of concept ✅
2. **Long context:** Увеличить seq_len до 4K → 8K → 16K → 32K
3. **QA датасет:** needle-in-a-haystack (как обсуждали ранее)
4. **Добавить слой 17:** Предвычислить вход для слоя 17, обучить второй TitansBlock
5. **End-to-end fine-tune:** Собрать все обученные Titans-блоки, сквозной FT

## Troubleshooting

### OOM при предвычислении
- Уменьшить `--batch_size` до 4 или 2
- Использовать `--dtype bfloat16`
- gemma.gm `return_hidden_states` может хранить все промежуточные — может понадобиться кастомный forward (см. комментарии в precompute_activations.py)

### Несовпадение размеров
- `max_seq_len` при предвычислении **должен совпадать** с `seq_len` при обучении
- `embed_dim` автоматически определяется из модели (1152 для Gemma 1B)

### Несовпадение порядка примеров
- Порядок шардов соответствует порядку в HF датасете
- Для CE loss нужен тот же датасет токенов (для targets)
- Shuffle активаций и токенов **одинаково** (см. `activations_dataloader.py`)
