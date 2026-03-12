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

  def transform(self, state: _StateT) -> _StateT:  # pytype: disable=signature-mismatch
    # Remove the Titans weights from the params structure so it can be restored
    original_params, titans_params = titans_tree_utils.split_titans_params(state.params)

    state = state.replace(params=original_params)

    state = self.wrapped.transform(state)

    # Restore the Titans weights
    state = state.replace(params=titans_tree_utils.merge_titans_params(state.params, titans_params))

    return state
