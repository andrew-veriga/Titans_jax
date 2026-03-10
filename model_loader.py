import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os

def stitch_hybrid_model(base_gemma_params, titans_delta_params):
    """
    Merges official Gemma weights with the trained Titans delta.
    
    Args:
        base_gemma_params: Pytree of the original Gemma weights.
        titans_delta_params: Pytree of the Titans modules (with None for base parts).
        
    Returns:
        A complete hybrid Pytree ready for model.apply()
    """
    def merge_fn(base, delta):
        # If we have a delta value (Titans weight), use it.
        # Otherwise, keep the base Gemma weight.
        if delta is not None:
            return delta
        return base

    return jax.tree_util.tree_map(merge_fn, base_gemma_params, titans_delta_params)

def load_titans_delta(path):
    """Loads the small Titans-only checkpoint."""
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(os.path.abspath(path))

# Example usage pattern for Colab:
# 1. Load official Gemma:
#    params_2b = gm.ckpts.load_from_path(CKPT_PATH)
# 2. Load Titans Delta:
#    delta = load_titans_delta("./titans_delta_init")
# 3. Stitch:
#    full_hybrid_params = stitch_hybrid_model(params_2b, delta)
