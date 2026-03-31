import jax
import orbax.checkpoint as ocp
import os

# Загружаем сохранённые titans веса
titans_delta_path = "./saved_titans_delta"
if not os.path.exists(titans_delta_path):
    import zipfile
    with zipfile.ZipFile("saved_titans_delta.zip", "r") as z:
        z.extractall(titans_delta_path)

checkpointer = ocp.StandardCheckpointer()
params = checkpointer.restore(os.path.abspath(titans_delta_path))

# Вывод дерева параметров
print("\n" + "="*80)
print("TITANS ПАРАМЕТРЫ (memory + memory_gate)")
print("="*80)

paths_and_shapes = []
jax.tree_util.tree_map_with_path(
    lambda path, v: paths_and_shapes.append(
        ('/'.join(str(p.key) for p in path), v.shape, v.dtype, v.size)
    ),
    params
)

total = 0
for path, shape, dtype, size in sorted(paths_and_shapes):
    print(f"{path}: {shape} [{dtype}]")
    total += size

print("="*80)
print(f"Итого memory параметров: {total:,}")
print("="*80)
