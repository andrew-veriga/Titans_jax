import os
os.environ['KAULDRON_TYPECHECK'] = '0'
os.environ['KD_CHECK_TYPES'] = '0'

import sys
sys.path.insert(0, os.path.abspath('.'))

import jax
import jax.numpy as jnp
from gemma import gm
import importlib

import gemma_titans
importlib.reload(gemma_titans)
from gemma_titans import Gemma3_1B_Titans

# Инициализация модели
model = Gemma3_1B_Titans(
    config=Gemma3_1B_Titans.config,
    dtype=jnp.bfloat16,
    return_last_only=False,
    tokens="batch.tokens",
)

# Загрузка весов
print("Загрузка весов Gemma3 1B IT...")
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

# Вывод дерева параметров
print("\n" + "="*80)
print("ДЕРЕВО ПАРАМЕТРОВ")
print("="*80)

total_params = 0
memory_params = 0

paths_and_shapes = []
jax.tree_util.tree_map_with_path(
    lambda path, v: paths_and_shapes.append(
        ('/'.join(str(p.key) for p in path), v.shape, v.dtype, v.size)
    ),
    params
)

for path, shape, dtype, size in sorted(paths_and_shapes):
    is_memory = 'memory' in path or 'titans' in path.lower()
    marker = " <<<" if is_memory else ""
    print(f"{path}: {shape} [{dtype}]{marker}")
    total_params += size
    if is_memory:
        memory_params += size

print("="*80)
print(f"Всего параметров:   {total_params:,}")
print(f"Memory параметров:  {memory_params:,} ({100*memory_params/total_params:.2f}%)")
print("="*80)

# Отдельно — только memory-параметры
print("\nТОЛЬКО MEMORY ПАРАМЕТРЫ:")
print("-"*80)
for path, shape, dtype, size in sorted(paths_and_shapes):
    if 'memory' in path or 'titans' in path.lower():
        print(f"  {path}: {shape}")
