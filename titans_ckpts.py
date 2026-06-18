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
    if self.ignore_checkpoint_attn:
      # Inject dummy attn structure so that orbax doesn't fail when loading base Gemma checkpoint
      # Find a normal layer to copy the attn structure from
      ref_attn = None
      for key, layer_params in original_params.items():
        if 'layer_' in key and 'attn' in layer_params:
          ref_attn = layer_params['attn']
          break
      
      if ref_attn is not None:
        for key, layer_params in original_params.items():
          if 'layer_' in key and 'attn' not in layer_params:
            layer_params['attn'] = ref_attn
            injected_attn_layers.append(key)

    state = state.replace(params=original_params)

    state = self.wrapped.transform(state)

    if injected_attn_layers:
      loaded_params = dict(state.params)
      for key in injected_attn_layers:
        if 'attn' in loaded_params[key]:
          layer_params = dict(loaded_params[key])
          del layer_params['attn']
          loaded_params[key] = layer_params
      state = state.replace(params=loaded_params)

    # Restore the Titans weights
    state = state.replace(params=titans_tree_utils.merge_titans_params(state.params, titans_params))

    return state
