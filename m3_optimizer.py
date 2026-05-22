"""
M3 Optimizer — Multi-scale Momentum Muon (from Hope / Nested Learning paper).

Combines fast momentum (per-step) with slow momentum (updated every C steps),
aggregated via Newton-Schulz spectral normalization + weighted sum.

Concept: M3 = Adam + Muon + CMS applied to optimizer momentum.
  - M_fast: updated every step (captures recent gradient landscape)
  - M_slow: updated every `slow_update_freq` steps (compressed summary of past landscape)
  - Aggregation: NS(M_fast) + w * NS(M_slow)

This prevents optimizer catastrophic forgetting when training on heterogeneous chunks.

Usage:
    from m3_optimizer import m3_optimizer
    opt = m3_optimizer(learning_rate=1e-3, slow_update_freq=16, slow_weight=0.1)

References:
    - Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025)
    - Wiki: concepts/Nested Learning and Hope.md
"""

from typing import Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
import optax
from optax._src import base


# ── Newton-Schulz 5-iteration (scalar-friendly, for 2D matrices) ──────────────

def _newton_schulz_5(x: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """Newton-Schulz spectral normalization, 5 iterations. Works on 2D matrices."""
    if x.ndim != 2:
        return x  # skip non-matrices

    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    # Aggressive coefficients from Muon/Titans-PyTorch
    a, b, c = 3.4445, -4.7750, 2.0315

    norm = jnp.linalg.norm(x, ord='fro')
    x = x / jnp.maximum(norm, eps)

    for _ in range(5):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x

    if transposed:
        x = x.T
    return x


# ── Newton-Schulz vectorized for parameter trees ─────────────────────────────

def _ns_tree(tree):
    """Apply NS normalization to all 2D matrix parameters in a tree."""
    def _apply(p):
        if p.ndim == 2:
            return _newton_schulz_5(p.astype(jnp.float32)).astype(p.dtype)
        return p
    return jax.tree_util.tree_map(_apply, tree)


# ── M3 State ──────────────────────────────────────────────────────────────────

class M3State(NamedTuple):
    """State for M3 optimizer."""
    count: jnp.ndarray          # step counter
    mu_fast: base.Params        # fast momentum (updated every step)
    mu_slow: base.Params        # slow momentum (updated every slow_update_freq steps)
    # TODO: add v_fast/v_slow for Adam-style second moment if needed
    # TODO: consider mu_dtype for memory efficiency (bfloat16 momenta)


# ── M3 Optimizer ──────────────────────────────────────────────────────────────

def m3_optimizer(
    learning_rate: base.ScalarOrSchedule,
    beta_fast: float = 0.9,
    beta_slow: float = 0.99,
    slow_update_freq: int = 16,
    slow_weight: float = 0.1,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """
    M3: Multi-scale Momentum with Newton-Schulz aggregation.

    Args:
        learning_rate: Learning rate or schedule.
        beta_fast: Momentum coefficient for fast scale (like Adam b1).
        beta_slow: Momentum coefficient for slow scale (closer to 1.0 for longer memory).
        slow_update_freq: Update slow momentum every N steps.
        slow_weight: Weight for slow momentum in aggregation (0 = pure fast, 1 = pure slow).
        eps: Epsilon for numerical stability.
        mu_dtype: Optional dtype for momentum storage.
        weight_decay: Optional decoupled weight decay.

    Returns:
        optax.GradientTransformation with M3 logic.
    """
    # TODO: validate that slow_update_freq > 1 (otherwise degrades to standard momentum)
    # TODO: add option to use Adam second moment (v) alongside momentum (mu)
    # TODO: add warmup for slow_weight (start from 0, ramp up)

    def init_fn(params):
        _zeros = lambda p: jnp.zeros_like(p, dtype=mu_dtype if mu_dtype is not None else p.dtype)
        mu_fast = jax.tree_util.tree_map(_zeros, params)
        mu_slow = jax.tree_util.tree_map(_zeros, params)
        count = jnp.zeros([], jnp.int32)
        return M3State(count=count, mu_fast=mu_fast, mu_slow=mu_slow)

    def update_fn(updates, state, params=None):
        lr = learning_rate(state.count) if callable(learning_rate) else learning_rate

        # --- Fast momentum: standard EMA ---
        # TODO: consider using Delta Momentum here (Phase 2) for content-dependent decay
        mu_fast = jax.tree_util.tree_map(
            lambda m, g: beta_fast * m + (1.0 - beta_fast) * g,
            state.mu_fast, updates
        )

        # --- Slow momentum: update only every slow_update_freq steps ---
        # Accumulate gradients between slow updates
        # When count % slow_update_freq == 0: mu_slow = beta_slow * mu_slow + (1 - beta_slow) * mean_grads
        # Otherwise: mu_slow unchanged
        is_slow_step = (state.count % slow_update_freq == 0) & (state.count > 0)

        mu_slow = jax.tree_util.tree_map(
            lambda m_slow, g: jnp.where(
                is_slow_step,
                beta_slow * m_slow + (1.0 - beta_slow) * g,
                m_slow
            ),
            state.mu_slow, updates
        )

        # --- Newton-Schulz normalization on both scales ---
        # TODO: NS is expensive for large matrices — consider skipping for small params
        # TODO: consider applying NS only to 2D kernels, skipping biases/norms
        ns_fast = _ns_tree(mu_fast)
        ns_slow = _ns_tree(mu_slow)

        # --- Aggregation: weighted sum ---
        new_updates = jax.tree_util.tree_map(
            lambda f, s: f + slow_weight * s,
            ns_fast, ns_slow
        )

        # --- Scale by learning rate ---
        new_updates = jax.tree_util.tree_map(
            lambda u: -lr * u, new_updates
        )

        # --- Optional weight decay ---
        if weight_decay > 0.0 and params is not None:
            new_updates = jax.tree_util.tree_map(
                lambda u, p: u - weight_decay * lr * p,
                new_updates, params
            )

        count = state.count + 1
        new_state = M3State(count=count, mu_fast=mu_fast, mu_slow=mu_slow)

        return new_updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


# ── Convenience: M3-Adam hybrid ───────────────────────────────────────────────

def m3_adam(
    learning_rate: base.ScalarOrSchedule,
    b1_fast: float = 0.9,
    b2: float = 0.999,
    b1_slow: float = 0.99,
    slow_update_freq: int = 16,
    slow_weight: float = 0.1,
    eps: float = 1e-8,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    """
    M3-Adam: Adam with multi-scale first moment.

    Uses Adam's second moment (v) for adaptive learning rates,
    but M3's multi-scale first moment for direction.

    Args:
        learning_rate: Learning rate or schedule.
        b1_fast: Fast momentum beta (Adam b1 default).
        b2: Adam second moment beta.
        b1_slow: Slow momentum beta.
        slow_update_freq: Slow momentum update frequency.
        slow_weight: Weight of slow momentum.
        eps: Adam epsilon.
        mu_dtype: Optional dtype for momenta.
    """
    # TODO: implement Adam second moment integration with M3 first moment
    # For now, delegates to m3_optimizer which is momentum-only (like Muon)
    return m3_optimizer(
        learning_rate=learning_rate,
        beta_fast=b1_fast,
        beta_slow=b1_slow,
        slow_update_freq=slow_update_freq,
        slow_weight=slow_weight,
        eps=eps,
        mu_dtype=mu_dtype,
    )