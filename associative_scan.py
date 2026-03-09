from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array

def pad_at_dim(t: Array, pad: tuple[int, int], dim: int = -1, value: float = 0.):
    """
    Pads a tensor at a specified dimension.
    """
    dims = t.ndim
    if dim < 0:
        dim = dims + dim
    
    pad_width = [(0, 0)] * dims
    pad_width[dim] = pad
    return jnp.pad(t, pad_width, mode='constant', constant_values=value)

def binary_operator(
    a: tuple[Array, Array],
    b: tuple[Array, Array]
):
    """
    Binary operator used in the associative_scan function.
    """
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, kv_j + a_j * kv_i

def associative_scan(
    operator: Callable,
    elems: tuple[Array, Array]
):
    """
    Performs an associative scan on the input tuples.
    Uses JAX's lax.associative_scan.
    """
    # JAX's lax.associative_scan scans over the first dimension (axis 0).
    # Since our input is (batch, seq, ...), we vmap over the batch dimension.
    return jax.vmap(lambda x: jax.lax.associative_scan(operator, x))(elems)
