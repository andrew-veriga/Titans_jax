"""
Titans Memory optimizer with M3 routing:
  m3_optimizer for projections / Adam-atan2 for gates / Adam-atan2 for base.

M3 (Multi-scale Momentum) replaces Muon, adding slow momentum that updates
every `slow_update_freq` steps, aggregated via Newton-Schulz normalization.
This prevents optimizer catastrophic forgetting on heterogeneous chunks.

Usage:
    from routing_optimizer import make_routing_optimizer

    routing_optimizer = make_routing_optimizer(opt_params)

where opt_params is a dict with keys:
    lr_muon, beta, lr_adam, adam_b1, adam_b2,
    lr_gate, gate_b1, gate_b2, every_k_schedule,
    slow_update_freq (optional, default 16),
    slow_weight (optional, default 0.1)
"""

import jax
import jax.numpy as jnp
import optax
from kauldron import kd
from adam_atan2 import adam_atan2
from m3_optimizer import m3_optimizer


# ── mask helpers ─────────────────────────────────────────────────────────────

M3_KEYS = {"to_queries", "to_keys_values", "combine_heads"}


def is_m3_param(path_str: str) -> bool:
    """True for attention projection kernels (routed to M3 optimizer)."""
    parts = path_str.split("/")
    return (
        len(parts) >= 2
        and parts[-1] == "kernel"
        and parts[-2] in M3_KEYS
    )


def m3_mask(params):
    """Mask for parameters that use M3 optimizer (attention projections)."""
    def _m(path, v):
        return is_m3_param("/".join(str(p.key) for p in path))
    return jax.tree_util.tree_map_with_path(_m, params)


def is_gate_param(path_str: str) -> bool:
    return "memory_gate_proj" in path_str.split("/")


def gate_mask(params):
    def _m(path, v):
        return is_gate_param("/".join(str(p.key) for p in path))
    return jax.tree_util.tree_map_with_path(_m, params)


def adam_base_mask(params):
    def _m(path, v):
        path_str = "/".join(str(p.key) for p in path)
        return not is_m3_param(path_str) and not is_gate_param(path_str)
    return jax.tree_util.tree_map_with_path(_m, params)


# ── public API ───────────────────────────────────────────────────────────────


def make_routing_optimizer(opt_params: dict):
    """Build a 3-way routed optimizer wrapped in ``partial_updates`` + ``MultiSteps``.

    Routing:
        1. **M3 optimizer** (multi-scale momentum + Newton-Schulz) for attention
           projection kernels: to_queries, to_keys_values, combine_heads.
        2. **Adam-atan2** for memory gate parameters (memory_gate_proj).
        3. **Adam-atan2** for remaining memory parameters.

    Args:
        opt_params: dict with keys
            lr_muon           – learning-rate schedule or float for M3 (attention projections)
            beta              – M3 fast momentum beta
            lr_adam           – learning-rate schedule or float for Adam (base memory params)
            adam_b1           – Adam b1 schedule or float (base)
            adam_b2           – Adam b2 float (base)
            lr_gate           – learning-rate schedule or float for Adam (gate params)
            gate_b1           – Adam b1 schedule or float (gate)
            gate_b2           – Adam b2 float (gate)
            every_k_schedule  – int, gradient accumulation / update frequency
            slow_update_freq  – int, M3 slow momentum update frequency (default 16)
            slow_weight       – float, M3 slow momentum weight (default 0.1)

    Returns:
        An ``optax.GradientTransformation`` ready to pass to ``kd.train.Trainer``.
    """
    slow_freq = opt_params.get("slow_update_freq", 16)
    slow_wt = opt_params.get("slow_weight", 0.1)
    every_k = opt_params["every_k_schedule"]

    # CMS principle: slow momentum must update at most as often as the optimizer
    # steps.  M3 counts in optimizer steps (inside MultiSteps), so slow_update_freq
    # must be >= every_k_schedule.  Auto-correct with a warning if violated.
    if slow_freq < every_k:
        import warnings
        warnings.warn(
            f"routing_optimizer: slow_update_freq ({slow_freq}) < "
            f"every_k_schedule ({every_k}).  "
            f"Auto-correcting slow_update_freq to {every_k}.",
            stacklevel=2,
        )
        slow_freq = every_k

    inner_chain = optax.chain(
        optax.clip_by_global_norm(1.0),
        # 1. M3 for attention projections (replaces Muon)
        optax.masked(
            m3_optimizer(
                learning_rate=opt_params["lr_muon"],
                beta_fast=opt_params["beta"],
                beta_slow=0.99,
                slow_update_freq=slow_freq,
                slow_weight=slow_wt,
                eps=1e-8,
                mu_dtype=jnp.float32,
            ),
            mask=m3_mask,
        ),
        # 2. Adam-atan2 for memory gates (higher LR)
        optax.masked(
            adam_atan2(
                learning_rate=opt_params["lr_gate"],
                b1=opt_params["gate_b1"],
                b2=opt_params["gate_b2"],
                eps=1e-8,
                mu_dtype=jnp.float32,
            ),
            mask=gate_mask,
        ),
        # 3. Adam-atan2 for remaining memory params
        optax.masked(
            adam_atan2(
                learning_rate=opt_params["lr_adam"],
                b1=opt_params["adam_b1"],
                b2=opt_params["adam_b2"],
                eps=1e-8,
                mu_dtype=jnp.float32,
            ),
            mask=adam_base_mask,
        ),
    )

    return optax.MultiSteps(
        kd.optim.partial_updates(
            inner_chain,
            mask=kd.optim.select(["memory", "memory_gate_proj"]),
        ),
        every_k_schedule=opt_params["every_k_schedule"],
    )