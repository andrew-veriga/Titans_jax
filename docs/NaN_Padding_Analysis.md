# Анализ: NaN от padding-токенов в Newton-Schulz нормализации

## Диагноз разработчика — ВЕРНЫЙ ✅

Логика полностью корректна. Вот детальный разбор:

---

## Цепочка событий, приводящая к NaN

### 1. Источник нулей: `precompute_activations.py` (строки 568–570)
```python
mask_expanded = batch_masks[:, :, None].astype(save_dtype)  # (B, L, 1)
hidden_np = hidden_np * mask_expanded
```
Padding-токены (позиции где `mask=0`) получают **строго нулевые активации** `=0.0`.

### 2. Нули проходят через Titans Block
```
zero_activations → store_norm(seq) → to_keys_values(kv_seq) → keys=0, values=0
```
Все проекции `Dense(use_bias=False)` — без байаса → `0 @ W = 0`. Ключи и значения **строго нулевые**.

### 3. Градиенты памяти тоже нулевые
`forward_and_loss` в `scan_step` (строки 436):
- `pred = memory_model(past_weights, keys=0)` → `pred ≈ 0` (зависит от весов, но при нулевом входе)
- `target = values = 0` → `loss = MSE(0, 0) = 0`
- **Градиент** `g = ∂loss/∂weights` = **строго нулевая матрица** для padding-позиций

### 4. Катастрофа в Newton-Schulz: `apply_fast_ns_to_tensor`
Градиенты (нулевые) → negated (всё ещё нулевые) → `apply_fast_ns_to_tensor`:

**БЫЛО (до исправления):**
```python
norm = jnp.linalg.norm(t_3d)  # √(Σx²) = √0 = 0
t_3d = t_3d / norm             # 0/0 → NaN!
```

При обратном проходе (meta-градиенты): `d/dx[√(Σx²)]` при `x=0` → **деление на ноль** → NaN.

**СТАЛО (исправление, строки 201–205):**
```python
sq_norm = jnp.sum(jnp.square(t_3d), axis=(-2, -1), keepdims=True)
norm = jnp.sqrt(sq_norm + 1e-12)  # √(0 + 1e-12) ≈ 1e-6
t_3d = t_3d / norm                 # 0 / 1e-6 = 0  ✅ безопасно
```

`1e-12` добавлен **под корень** (а не после), что гарантирует:
- При `x=0`: `√(0+1e-12) ≈ 1e-6` → `0/1e-6 = 0` — градиент конечный
- При `x≠0`: `1e-12` пренебрежимо мала — не влияет на результат

### 5. Распространение NaN по батчу
```
NaN в padding-позиции → associative_scan(binary_operator) → заражает весь батч
→ optax.zero_nans() → обнуляет ВСЁ обновление → обучение заморожено
```

---

## Оценка исправления: CORRECT ✅

Исправление в `apply_fast_ns_to_tensor` (строки 201–205) — **правильное и достаточное** для основной траектории. Дополнительно:

### Многоуровневая защита (уже в коде):
1. **Guard 1** (строки 431–432): `jnp.clip(k, -10, 10)` — ограничивает входы в grad_fn
2. **Guard 2** (строки 439–441): `nan_to_num(g, nan=0.0)` — зануляет NaN в градиентах **до** NS нормализации
3. **Safe norm** (строки 203–204): `√(sq + 1e-12)` — предотвращает NaN в самом NS
4. **Clip** (строка 215): `jnp.clip(t_3d, -1e3, 1e3)` — промежуточная защита внутри NS итераций
5. **Clip** (строка 486): `jnp.clip(surprises, -5.0, 5.0)` — перед ассоциативным сканом

---

## ⚠️ Оставшиеся риски (НЕ в основной траектории)

### Риск 1: `newton_schulz_norm_matrix` (строка 159) — СРЕДНИЙ
```python
norm = jnp.linalg.norm(x, ord='fro')
x_scaled = x / (norm + eps)  # eps=1e-7 ПОСЛЕ корня — НЕ безопасно для градиентов!
```
**Проблема:** `eps` добавлен после `linalg.norm`, а не под корень. Производная `d/dx[linalg.norm(x)]` при `x=0` всё ещё NaN. `norm + eps` не спасает — NaN уже вычислен внутри `linalg.norm`.

**Но:** Эта функция вызывается только через `apply_ns_to_tensor` (строка 237), которая **НЕ используется** в текущем коде — на строке 483 вызывается `apply_fast_ns_to_tensor`. Так что это **спящий риск**.

**Рекомендация:** Если `newton_schulz_norm_matrix` понадобится в будущем, заменить:
```python
# БЫЛО:
norm = jnp.linalg.norm(x, ord='fro')
x_scaled = x / (norm + eps)

# НАДО:
sq_norm = jnp.sum(jnp.square(x))
norm = jnp.sqrt(sq_norm + 1e-12)
x_scaled = x / norm
```

### Риск 2: `softclamp_grad_norm` (строка 63) — НИЗКИЙ
```python
norm = jnp.linalg.norm(t, axis=-1, keepdims=True)
t = t * (clamped_norm / jnp.maximum(norm, 1e-12))
```
**Проблема:** `jnp.linalg.norm` при нулевом тензоре → NaN в градиенте. `jnp.maximum(norm, 1e-12)` в знаменателе не спасает, т.к. NaN уже в числителе (`clamped_norm` содержит NaN от `softclamp_max(norm, ...)` при `norm=0`).

**Но:** `softclamp_grad_norm` вызывается только если `max_grad_norm` задан (строка 473):
```python
if exists(self.max_grad_norm):
    grads = jax.tree_util.tree_map(lambda t: softclamp_grad_norm(t, self.max_grad_norm), grads)
```
По умолчанию `max_grad_norm = None` → функция **не вызывается**. Кроме того, Guard 2 (`nan_to_num`) к этому моменту уже очистил бы нулевые градиенты от padding, так что `norm > 0` для реальных токенов.

**Рекомендация:** Если `max_grad_norm` будет включён, заменить `linalg.norm` на безопасную версию.

---

## Итоговая таблица

| Точка NaN-риска | Статус | Путь вызова |
|---|---|---|
| `apply_fast_ns_to_tensor` (стр. 203) | ✅ **Исправлено** | Основной путь, строка 483 |
| Guard 2 `nan_to_num` (стр. 440) | ✅ Дополнительная защита | Основной путь, строка 440 |
| `newton_schulz_norm_matrix` (стр. 159) | ⚠️ Не исправлено, но **не используется** | `apply_ns_to_tensor` → не вызывается |
| `softclamp_grad_norm` (стр. 63) | ⚠️ Не исправлено, но **max_grad_norm=None** | Только если включён `max_grad_norm` |

**Вердикт:** Исправление разработчика корректно и полностью решает проблему NaN от padding-токенов в текущей конфигурации обучения.