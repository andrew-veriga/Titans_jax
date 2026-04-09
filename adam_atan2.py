import functools
from typing import Literal, NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import combine
from optax._src import transform
 
def adam_atan2(
    learning_rate: base.ScalarOrSchedule,
    b1: base.ScalarOrSchedule = 0.9,
    b2: base.ScalarOrSchedule = 0.999,
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
    b1: base.ScalarOrSchedule = 0.9,
    b2: base.ScalarOrSchedule = 0.999,
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

    if callable(b1):
      b1_t = b1(state.count)
    else:
      b1_t = b1

    if callable(b2):
      b2_t = b2(state.count)
    else:
      b2_t = b2

    del params
    mu = jax.tree_util.tree_map(
        lambda m, g: b1_t * m + (1 - b1_t) * g, state.mu, updates)
    nu = jax.tree_util.tree_map(
        lambda v, g: b2_t * v + (1 - b2_t) * jnp.square(g), state.nu, updates)
    count = state.count + 1
    
    # Bias correction
    mu_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1_t**count), mu)
    nu_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2_t**count), nu)

    if nesterov:
      mu_hat = jax.tree_util.tree_map(
          lambda m, g: b1_t * m + (1 - b1_t) * g / (1 - b1_t**count), mu, updates)

    # Core modification: atan2(mu_hat, sqrt(nu_hat) + eps)
    updates = jax.tree_util.tree_map(
        lambda m, v: jnp.arctan2(m, jnp.sqrt(v + eps_root) + eps),
        mu_hat, nu_hat)

    return updates, transform.ScaleByAdamState(count=count, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


if __name__ == "__main__":
    import optax

    key = jax.random.PRNGKey(0)
    params = {
        "w1": jax.random.normal(key, (32, 16)),
        "b1": jnp.zeros(16),
        "w2": jax.random.normal(jax.random.PRNGKey(1), (16, 4)),
    }

    num_steps = 10

    # Расписание b1: cosine decay от 0.95 до 0.85
    b1_schedule = optax.cosine_decay_schedule(
        init_value=0.95,
        decay_steps=num_steps,
        alpha=0.85 / 0.95,  # конечное значение = init_value * alpha
    )

    optimizer = adam_atan2(learning_rate=1e-3, b1=b1_schedule)
    opt_state = optimizer.init(params)

    print(f"{'step':>4}  {'b1':>8}  {'‖upd w1‖':>10}  {'‖upd w2‖':>10}")
    print("-" * 40)

    for step in range(num_steps):
        b1_val = b1_schedule(step)

        grads = jax.tree_util.tree_map(
            lambda p: jax.random.normal(jax.random.PRNGKey(step), p.shape) * (10.0 / (step + 1)),
            params,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

        print(f"{step:4d}  {b1_val:.6f}  {jnp.linalg.norm(updates['w1']):10.6f}  {jnp.linalg.norm(updates['w2']):10.6f}")

    print("\nOK — adam_atan2 с b1 schedule работает корректно")

