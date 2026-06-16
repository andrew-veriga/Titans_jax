## Фикс: loss=28.8 не снижается — три изменения

### 1. `gemma_titans.py` — переключение loss с `cos_by_softmax` на `normalized_mse`

**Проблема:** `cos_by_softmax` вычисляет cross-entropy между softmax-распределениями delta_teacher и delta_student по D=1152 фичам. При D=1152 softmax сильно расплющивает распределение → градиент ≈ `1/1152 - 1/1152` ≈ 0. Градиент к memory params через длинную цепочку (`gate * retrieved → memory MLP`) становится ничтожным.

**Фикс:** Переключили на `normalized_mse` (косинусное расстояние):
```python
# Было: layer_loss = self.cos_by_softmax(delta_teacher, delta_student)
# Стало:
layer_loss = self.normalized_mse(delta_teacher, delta_student)
```
`normalized_mse` вычисляет `||norm(student) - norm(teacher)||²` — прямое сравнение без softmax-разрушения градиентов.

### 2. `gemma_titans.py` — gate `bias_init` -1.0 → 0.0

sigmoid(-1)=0.27 слишком сильно масштабирует градиенты к memory params (на 0.27). С `bias_init=0.0` (sigmoid=0.5) сбалансированный старт: memory вносит 50%, local_attn (Gemma) — 50%.

### 3. `colabs/Titans_jax_Phase1_training.ipynb` — `lr_adam` 1e-4 → 3e-4

Memory architecture params (key/value/query projections) учатся с нуля. 1e-4 с `every_k_schedule=4` (MultiSteps) — слишком консервативно для from-scratch обучения.

### Ожидаемый результат
- `normalized_mse` даёт сильные градиенты к memory params даже при D=1152
- Gate=0.5 не масштабирует градиенты → memory учится быстрее
- lr_adam=3e-4 ускоряет обучение в 3 раза