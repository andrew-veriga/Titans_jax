import jax
import jax.numpy as jnp
from typing import Callable, Tuple

def conjugate_gradient_solve(
    H_apply_fn: Callable[[jnp.ndarray], jnp.ndarray], 
    q: jnp.ndarray, 
    x0: jnp.ndarray, 
    max_iter: int = 15, 
    tol: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Conjugate Gradient (CG) solver tailored for MesaNet's Test-Time Training.
    
    In MesaNet, we want to find the optimal fast weights `x` by solving the 
    linear system: (H_t + \Lambda) * x_t = q_t
    
    Instead of explicitly maintaining the d x d matrix H_t, we use an implicit 
    function `H_apply_fn` that computes the matrix-vector product.
    
    Args:
        H_apply_fn: A function that computes `(H + \Lambda) @ p`. 
                    In MesaNet, this is highly optimized using hardware-efficient 
                    Gated Linear Attention (GLA) structures.
        q: The query vector (right-hand side of the equation). Shape: [..., d]
        x0: Initial guess for `x`. Usually zeros or the weights from the previous step. Shape: [..., d]
        max_iter: Maximum number of iterations (the "dynamic compute" budget).
        tol: Tolerance for the squared residual. If the error drops below this, 
             the solver stops early, saving compute.
             
    Returns:
        x: The optimized weights for the current step/chunk.
        num_iters: The actual number of iterations executed.
    """
    # r = b - A*x
    r0 = q - H_apply_fn(x0)
    p0 = r0
    
    # Squared residual (r^T r)
    rs_old0 = jnp.sum(r0 * r0, axis=-1, keepdims=True)
    
    def cond_fun(state):
        i, x, r, p, rs_old = state
        # Continue if we haven't hit max_iter AND the residual is still above tolerance
        # (Using jnp.max to handle batched execution safely)
        return jnp.logical_and(i < max_iter, jnp.max(rs_old) > tol)

    def body_fun(state):
        i, x, r, p, rs_old = state
        
        # A * p step (The most computationally intensive part)
        Ap = H_apply_fn(p)
        
        # alpha = (r^T r) / (p^T A p)
        pAp = jnp.sum(p * Ap, axis=-1, keepdims=True)
        # Avoid division by zero
        alpha = jnp.where(pAp > 1e-8, rs_old / pAp, 0.0)
        
        # Update weights x and residual r
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = jnp.sum(r * r, axis=-1, keepdims=True)
        
        # beta = (r_new^T r_new) / (r_old^T r_old)
        beta = jnp.where(rs_old > 1e-8, rs_new / rs_old, 0.0)
        
        # Update search direction p
        p = r + beta * p
        
        return i + 1, x, r, p, rs_new

    # Initialize the while loop state
    initial_state = (jnp.array(0), x0, r0, p0, rs_old0)
    
    # jax.lax.while_loop enables dynamic loop unrolling, crucial for the 
    # adaptive test-time compute feature of MesaNet.
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    
    final_i, final_x, _, _, _ = final_state
    
    return final_x, final_i
