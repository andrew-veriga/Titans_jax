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
    def _merge(base, delta):
        # Handle dictionary-like structures
        if hasattr(base, 'items') and hasattr(delta, 'items'):
            merged = dict(base)
            for k, v in delta.items():
                if k in merged:
                    merged[k] = _merge(merged[k], v)
                else:
                    merged[k] = v
            # If the original base was a specific class like FrozenDict, 
            # returning a standard dict is generally fine for JAX/Flax.
            return merged
        # Handle lists or tuples
        elif isinstance(base, (list, tuple)) and isinstance(delta, (list, tuple)):
            merged = []
            for b, d in zip(base, delta):
                merged.append(_merge(b, d))
            return type(base)(merged)
        else:
            # For leaves, if delta has a value use it, else keep base weight
            return delta if delta is not None else base

    return _merge(base_gemma_params, titans_delta_params)

def load_titans_delta(path):
    """Loads the small Titans-only checkpoint."""
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(os.path.abspath(path))

# Example usage pattern for Colab:
# 1. Load official Gemma:
#    params_2b = gm.ckpts.load_params(CKPT_PATH)
# 2. Load Titans Delta:
#    delta = load_titans_delta("./titans_delta_init")
# 3. Stitch:
#    full_hybrid_params = stitch_hybrid_model(params_2b, delta)
