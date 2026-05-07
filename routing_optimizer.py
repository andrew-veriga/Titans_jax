"""
Titans Memory optimizer with 3-way routing: Muon / Adam-atan2 for gates / Adam-atan2 for base.

Usage:
    from routing_optimizer import make_routing_optimizer

    routing_optimizer = make_routing_optimizer(opt_params)

where opt_params is a dict with keys:
    lr_muon, beta, lr_adam, adam_b1, adam_b2,
    lr_gate, gate_b1, gate_b2, every_k_schedule
"""

import jax
import jax.numpy as jnp
import optax
from kauldron import kd
from adam_atan2 import adam_atan2
from optax.contrib._muon import MuonDimensionNumbers


# ── mask helpers ─────────────────────────────────────────────────────────────

MUON_KEYS = {"to_queries", "to_keys_values", "combine_heads"}


def muon_only_dims(params):
    """Return MuonDimensionNumbers for attention projection kernels."""

    def _label(path, v):
        key = str(path[-1].key) if hasattr(path[-1], "key") else ""
        parent = str(path[-2].key) if len(path) > 1 and hasattr(path[-2], "key") else ""
        if key == "kernel" and parent in MUON_KEYS:
            return MuonDimensionNumbers(reduction_axis=0, output_axis=1)
        return None

    return jax.tree_util.tree_map_with_path(_label, params)


def is_muon_param(path_str: str) -> bool:
    parts = path_str.split("/")
    return (
        len(parts) >= 2
        and parts[-1] == "kernel"
        and parts[-2] in MUON_KEYS
    )


def muon_mask(params):
    def _m(path, v):
        return is_muon_param("/".join(str(p.key) for p in path))

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
        return not is_muon_param(path_str) and not is_gate_param(path_str)

    return jax.tree_util.tree_map_with_path(_m, params)


# ── public API ───────────────────────────────────────────────────────────────


def make_routing_optimizer(opt_params: dict):
    """Build a 3-way routed optimizer wrapped in ``partial_updates`` + ``MultiSteps``.

    Args:
        opt_params: dict with keys
            lr_muon           – learning-rate schedule or float for Muon (attention projections)
            beta              – Muon momentum beta
            lr_adam           – learning-rate schedule or float for Adam (base memory params)
            adam_b1           – Adam b1 schedule or float (base)
            adam_b2           – Adam b2 float (base)
            lr_gate           – learning-rate schedule or float for Adam (gate params)
            gate_b1           – Adam b1 schedule or float (gate)
            gate_b2           – Adam b2 float (gate)
            every_k_schedule  – int, gradient accumulation / update frequency

    Returns:
        An ``optax.GradientTransformation`` ready to pass to ``kd.train.Trainer``.
    """
    inner_chain = optax.chain(
        optax.clip_by_global_norm(1.0),
        # 1. Muon for attention projections
        optax.masked(
            optax.contrib.muon(
                learning_rate=opt_params["lr_muon"],
                muon_weight_dimension_numbers=muon_only_dims,
                beta=opt_params["beta"],
                eps=1e-8,
                mu_dtype=jnp.float32,
            ),
            mask=muon_mask,
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