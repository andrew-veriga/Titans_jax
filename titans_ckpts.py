import dataclasses
import typing
from typing import Any, TypeVar, Union

from kauldron import kd
import titans_tree_utils

if typing.TYPE_CHECKING:
  # Likely overkill, but avoid resolving the lazy-import on importing this file.
  _StateT = TypeVar('_StateT', bound=kd.train.TrainState)
else:
  _StateT = TypeVar('_StateT')


@dataclasses.dataclass(frozen=True)
class SkipTitans(kd.ckpts.PartialKauldronLoader):
  """Wraps a partial loader to not restore the Titans weights."""

  wrapped: kd.ckpts.PartialKauldronLoader
  ignore_checkpoint_attn: bool = True

  def transform(self, state: _StateT) -> _StateT:  # pytype: disable=signature-mismatch
    # Remove the Titans weights from the params structure so it can be restored
    original_params, titans_params = titans_tree_utils.split_titans_params(state.params)

    injected_attn_layers = []
    injected_ffn_layers = []

    # ── Collect reference structures from a non-Titans Gemma layer ──
    ref_attn = None
    ref_ffn = {}  # {'mlp': ..., 'pre_ffw_norm': ..., 'post_ffw_norm': ...}
    for key, layer_params in original_params.items():
      if 'layer_' in key and isinstance(layer_params, dict):
        if 'attn' in layer_params and ref_attn is None:
          ref_attn = layer_params['attn']
        for ffn_key in ('mlp', 'pre_ffw_norm', 'post_ffw_norm'):
          if ffn_key in layer_params and ffn_key not in ref_ffn:
            ref_ffn[ffn_key] = layer_params[ffn_key]

    # ── Inject dummy structures so Orbax can match the Gemma checkpoint ──
    for key, layer_params in original_params.items():
      if 'layer_' in key and isinstance(layer_params, dict):
        # Inject attn if missing (Phase 2 Titans layers without attention)
        if self.ignore_checkpoint_attn and 'attn' not in layer_params and ref_attn is not None:
          layer_params['attn'] = ref_attn
          injected_attn_layers.append(key)
        # Inject FFN if missing (Titans layers use titans_ffn instead of mlp)
        if 'mlp' not in layer_params and ref_ffn:
          for ffn_key, ffn_val in ref_ffn.items():
            layer_params[ffn_key] = ffn_val
          injected_ffn_layers.append(key)

    state = state.replace(params=original_params)
    state = self.wrapped.transform(state)

    # ── Clean up injected dummy structures ──
    if injected_attn_layers or injected_ffn_layers:
      loaded_params = dict(state.params)
      for key in injected_attn_layers:
        if key in loaded_params and 'attn' in loaded_params[key]:
          layer_params = dict(loaded_params[key])
          del layer_params['attn']
          loaded_params[key] = layer_params
      for key in injected_ffn_layers:
        if key in loaded_params:
          layer_params = dict(loaded_params[key])
          for ffn_key in ('mlp', 'pre_ffw_norm', 'post_ffw_norm'):
            layer_params.pop(ffn_key, None)
          loaded_params[key] = layer_params
      state = state.replace(params=loaded_params)

    # Restore the Titans weights (titans_ffn, titans_pre_ffw_norm, etc.)
    state = state.replace(params=titans_tree_utils.merge_titans_params(state.params, titans_params))

    return state
