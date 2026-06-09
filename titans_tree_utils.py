from __future__ import annotations
import copy
from typing import Any, NamedTuple
from collections.abc import Mapping

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
    if isinstance(value, Mapping):
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

# Keys that belong to the Titans-specific parts of a TitansBlock
# (not present in the original Gemma Block).
_TITANS_KEYS = frozenset({
    'memory', 'memory_gate_proj',
    'titans_ffn', 'titans_pre_ffw_norm', 'titans_post_ffw_norm',
})

def split_titans_params(params: _ParamsDict) -> SplittedParams:
  """Split a nested tree into 2 trees, one with and without Titans-specific branches."""
  original_tree = {}
  titans_tree = {}

  def _split_recursive(input_subtree, original_subtree, titans_subtree):
    for key, value in input_subtree.items():
      if isinstance(value, Mapping):
        if key in _TITANS_KEYS:
          titans_subtree[key] = value
        else:
          original_subtree[key] = {}
          titans_subtree[key] = {}
          _split_recursive(value, original_subtree[key], titans_subtree[key])
      elif key in _TITANS_KEYS:
        titans_subtree[key] = value
      else:
        original_subtree[key] = value

  _split_recursive(params, original_tree, titans_tree)

  # Remove empty dicts in titans_tree
  def _remove_empty_dicts(tree):
    if not isinstance(tree, Mapping):
      return tree

    new_tree = {}
    for key, value in tree.items():
      if isinstance(value, Mapping):
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
          
      if isinstance(value, Mapping) and key in titans_subtree:
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
      if isinstance(layer_params, Mapping) and ('memory' in layer_params or 'memory_gate_proj' in layer_params):
        if 'attn' in layer_params:
          del layer_params['attn']

  # Initialize titans_* parameters from Gemma weights if not present in checkpoint.
  # This allows Titans FFN to start from pretrained Gemma weights instead of random init.
  _TITANS_FROM_GEMMA = {
      'mlp': 'titans_ffn',
      'pre_ffw_norm': 'titans_pre_ffw_norm',
      'post_ffw_norm': 'titans_post_ffw_norm',
  }
  for layer_name, layer_params in merged.items():
    if not isinstance(layer_params, Mapping):
      continue
    # Only process Titans layers (those with memory or memory_gate_proj)
    if 'memory' not in layer_params and 'memory_gate_proj' not in layer_params:
      continue
    for gemma_key, titans_key in _TITANS_FROM_GEMMA.items():
      if titans_key not in layer_params and gemma_key in layer_params:
        layer_params[titans_key] = copy.deepcopy(layer_params[gemma_key])
           
  return merged


def extract_and_merge_frozen_head(
    gemma_params: _ParamsDict,
    titans_params: _ParamsDict,
    after_layer: int,
    titans_layer_indices: list[int],
    remove_dead_attn: bool = True,
) -> _ParamsDict:
  """Extract frozen Gemma layers after ``after_layer`` and merge with trained Titans layers.

  Produces a parameter tree suitable for inference or further training that
  contains **only**:

  1. **Titans layers** (from ``titans_params``) for each index in
     ``titans_layer_indices``.  If the same layer key exists in
     ``gemma_params``, the two are merged with :func:`merge_titans_params`
     (Gemma base weights + Titans-specific weights).
  2. **Frozen Gemma layers** whose index is strictly greater than
     ``after_layer`` **and** not in ``titans_layer_indices``.
  3. **Non-layer params** (``final_norm``, ``embedder``, etc.) from
     ``gemma_params``.

  This is the inverse of "train only the TitansBlock on precomputed
  activations": after training you call this function to assemble a
  complete model checkpoint.

  Example::

      # After training TitansBlock on layer 23 with precomputed activations
      full_gemma = load_gemma_checkpoint(...)          # all 26 layers
      trained_titans = load_trained_checkpoint(...)     # layer_23 with memory

      assembled = extract_and_merge_frozen_head(
          gemma_params=full_gemma,
          titans_params=trained_titans,
          after_layer=22,                 # skip layers 0-22
          titans_layer_indices=[23],      # layer 23 is a Titans block
      )
      # assembled contains: layer_23 (merged), layer_24, layer_25,
      # final_norm, embedder

  Args:
      gemma_params: Full Gemma checkpoint parameter tree (inner dict
        with ``layer_0`` … ``layer_25``, ``final_norm``, ``embedder``).
      titans_params: Trained TitansBlock parameter tree.  Must have the
        same top-level ``layer_N`` keys as the layers it replaces.
        Typically the output of :func:`split_titans_params` or a saved
        training checkpoint.
      after_layer: Keep Gemma layers with index > this value as frozen.
        Layers with index ≤ this value are **dropped** (they were
        precomputed and are not needed).
      titans_layer_indices: Layer indices where trained Titans blocks
        should be placed.  These layers come from ``titans_params``
        (merged with Gemma base if available).
      remove_dead_attn: Forwarded to :func:`merge_titans_params`.  If
        True, removes unused ``attn`` weights from Titans layers.

  Returns:
      Assembled parameter dict with frozen head + Titans layers.
  """

  titans_set = set(titans_layer_indices)
  result: _ParamsDict = {}

  # --- 1. Non-layer params (final_norm, embedder, etc.) ---
  for key, value in gemma_params.items():
    if not key.startswith('layer_'):
      result[key] = value

  # --- 2. Titans layers (merge trained titans with gemma base) ---
  for idx in sorted(titans_set):
    layer_name = f'layer_{idx}'
    if layer_name in titans_params:
      if layer_name in gemma_params:
        # Merge: gemma base weights + titans-specific (memory, gate, etc.)
        result[layer_name] = merge_titans_params(
            gemma_params[layer_name],
            titans_params[layer_name],
            remove_dead_attn=remove_dead_attn,
        )
      else:
        # No gemma base — use titans params as-is
        result[layer_name] = titans_params[layer_name]

  # --- 3. Frozen Gemma layers after `after_layer` (not titans) ---
  for key, value in gemma_params.items():
    if not key.startswith('layer_'):
      continue
    try:
      idx = int(key.split('_', 1)[1])
    except (IndexError, ValueError):
      continue
    if idx > after_layer and idx not in titans_set:
      result[key] = value

  return result
