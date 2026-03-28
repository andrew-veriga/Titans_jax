import functools
from typing import Literal, NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import combine
from optax._src import transform
 
def adam_atan2(
    learning_rate: base.ScalarOrSchedule,
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
  """The Adam-atan2 optimizer.

  Adam-atan2 is a variant of Adam that uses the atan2 function to compute 
  the update direction, providing better stability by bounding the update 
  magnitude and improving robustness to gradient scale variations.
  """
  return combine.chain(
      scale_by_adam_atan2(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )

def scale_by_adam_atan2(
    b1: jax.typing.ArrayLike = 0.9,
    b2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
    eps_root: jax.typing.ArrayLike = 0.0,
    mu_dtype: Optional[Any] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """Rescale updates by adam-atan2 algorithm."""

  def init_fn(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=m, nu=v)

  def update_fn(updates, state, params=None):
    del params
    mu = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1 - b1) * g, state.mu, updates)
    nu = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1 - b2) * jnp.square(g), state.nu, updates)
    count = state.count + 1
    
    # Bias correction
    mu_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1**count), mu)
    nu_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2**count), nu)

    if nesterov:
      mu_hat = jax.tree_util.tree_map(
          lambda m, g: b1 * m + (1 - b1) * g / (1 - b1**count), mu, updates)

    # Core modification: atan2(mu_hat, sqrt(nu_hat) + eps)
    updates = jax.tree_util.tree_map(
        lambda m, v: jnp.arctan2(m, jnp.sqrt(v + eps_root) + eps),
        mu_hat, nu_hat)

    return updates, transform.ScaleByAdamState(count=count, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)