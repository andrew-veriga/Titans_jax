## Применён фикс: Убран `local_attn` из Phase 1 student path

### Причина неудачи предыдущих фиксов

`local_attn_output` — большой замороженный компонент в `delta_student`. `cos_by_softmax` применяет `softmax(delta)` по D=1152. Когда `local_attn` доминирует в `delta_student`, softmax насыщается: изменения в `gate*retrieved` почти не меняют распределение → **градиент к памяти ≈ 0**.

### Что изменилось в `gemma_titans.py`

Phase 1 student теперь использует **чистую память** (как в d3b801e):
```python
if self.use_original_attn:
    # Phase 1: PURE MEMORY — local_attn removed
    combined_output = gate * retrieved
else:
    # Phase 2: MAG hybrid — local_attn + memory
    combined_output = local_attn_output + gate * retrieved
```

### Сводка всех текущих фиксов
| Фикс | Статус |
|------|--------|
| ✅ Phase 1 student → чистая память (без local_attn) | Применён |
| ✅ Phase 1 → `cos_by_softmax` loss | Применён |
| ✅ `tanh(retrieved)` ограничение | Применён |
| ✅ `PRNGKey(0)` в `init_memory_state` | Применён |
| ✅ Фазо-зависимый FFN (`mlp` frozen в Phase 1) | Применён |
| ✅ `lr_adam=1e-4` | Применён |
| ✅ `lr_muon` восстановлен пользователем | Применён |

Запустите Phase 1 обучение (500-1000 шагов) и проверьте, снижается ли loss.