from __future__ import annotations
from typing import Any, NamedTuple

_ParamsDict = dict[str, Any]

class SplittedParams(NamedTuple):
  original: _ParamsDict
  titans: _ParamsDict

def migrate_static_gate_to_dynamic(params: _ParamsDict) -> _ParamsDict:
  """
  Removes the old static 'memory_gate' vector from the checkpoint parameters.
  This allows loading older Phase 1/2 checkpoints into the new architecture
  that uses a dynamic 'memory_gate_proj' Dense layer. The new layer will be
  initialized with random weights and requires a short re-training phase.
  """
  new_params = {}
  for key, value in params.items():
    if isinstance(value, dict):
      if key == 'memory_gate':
        print("Migrating checkpoint: Found and removed old static 'memory_gate'.")
        continue  # Skip adding this key
      else:
        new_params[key] = migrate_static_gate_to_dynamic(value)
    else:
      if key == 'memory_gate':
        print("Migrating checkpoint: Found and removed old static 'memory_gate'.")
        continue
      new_params[key] = value
  return new_params

def split_titans_params(params: _ParamsDict) -> SplittedParams:
  """Split a nested tree into 2 trees, one with and without 'memory' and 'memory_gate_proj' branches."""
  original_tree = {}
  titans_tree = {}

  def _split_recursive(input_subtree, original_subtree, titans_subtree):
    for key, value in input_subtree.items():
      if isinstance(value, dict):
        if key in ('memory', 'memory_gate_proj'):
          titans_subtree[key] = value
        else:
          original_subtree[key] = {}
          titans_subtree[key] = {}
          _split_recursive(value, original_subtree[key], titans_subtree[key])
      elif key in ('memory', 'memory_gate_proj'):
        titans_subtree[key] = value
      else:
        original_subtree[key] = value

  _split_recursive(params, original_tree, titans_tree)

  # Remove empty dicts in titans_tree
  def _remove_empty_dicts(tree):
    if not isinstance(tree, dict):
      return tree

    new_tree = {}
    for key, value in tree.items():
      if isinstance(value, dict):
        sub_tree = _remove_empty_dicts(value)
        if sub_tree:  # Only add if subtree is not empty
          new_tree[key] = sub_tree
      else:
        new_tree[key] = value
    return new_tree

  titans_tree = _remove_empty_dicts(titans_tree)

  return SplittedParams(original_tree, titans_tree)

def merge_titans_params(original: _ParamsDict, titans: _ParamsDict, remove_dead_attn: bool = False) -> _ParamsDict:
  """Inverse of `split_titans_params`.
  
  Args:
      remove_dead_attn: Если True, удаляет оригинальные веса 'attn' из слоев,
                        где присутствует 'memory' или 'memory_gate_proj'. Это экономит
                        память в архитектуре "Чистый Вариант Б", где оригинальное 
                        внимание Gemma не используется в слоях Titans.
  """

  def _merge_recursive(original_subtree, titans_subtree):
    new_tree = {}

    for key, value in original_subtree.items():
      # Пропускаем старый ключ даже если он есть в базе
      if key == 'memory_gate':
          continue
          
      if isinstance(value, dict) and key in titans_subtree:
        new_tree[key] = _merge_recursive(value, titans_subtree[key])
      else:
        new_tree[key] = value

    # Добавляем только новые ключи, ИГНОРИРУЯ старый memory_gate
    for k in sorted(set(titans_subtree) - set(original_subtree)):
      if k == 'memory_gate':
          continue
      new_tree[k] = titans_subtree[k]

    return new_tree

  merged = _merge_recursive(original, titans)
  
  if remove_dead_attn:
    for layer_name, layer_params in merged.items():
      if isinstance(layer_params, dict) and ('memory' in layer_params or 'memory_gate_proj' in layer_params):
        if 'attn' in layer_params:
          del layer_params['attn']
          
  return merged
