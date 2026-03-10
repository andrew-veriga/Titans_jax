import jax
import jax.numpy as jnp
from gemma.gm.nn import _config, _modules
from gemma_titans import GemmaTitansTransformer
import orbax.checkpoint as ocp
import os

def is_titans_param(path):
    """Check if a parameter path belongs to the Titans NLTM module."""
    path_str = "/".join([str(p.name) if hasattr(p, 'name') else str(p) for p in path])
    return 'memory' in path_str or 'memory_gate' in path_str

def perform_surgery_and_save_delta(output_path):
    # 1. Define the 2B config (or whichever version you are using)
    # This matches the architecture of the pretrained model
    config = _config.TransformerConfig(
        num_embed=256000,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=30.0,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attention_types=[_modules.AttentionType.GLOBAL] * 18, # Gemma-2B has 18 layers
    )
    
    model = GemmaTitansTransformer(config=config, dtype=jnp.float32)
    
    # 2. Initialize random parameters to get the structure
    print("Initializing hybrid structure...")
    rng = jax.random.PRNGKey(0)
    dummy_tokens = jnp.ones((1, 1), dtype=jnp.int32)
    variables = model.init(rng, tokens=dummy_tokens)
    params = variables['params']
    
    # 3. Extract only the Titans parameters (The Delta)
    # We use tree_util to walk the tree and keep only what we need
    def filter_titans(path, val):
        if is_titans_param(path):
            return val
        return None # Placeholder for non-titans weights

    print("Extracting Titans delta weights...")
    titans_delta = jax.tree_util.tree_map_with_path(filter_titans, params)
    
    # 4. Save the delta using Orbax
    print(f"Saving Titans delta to {output_path}...")
    checkpointer = ocp.StandardCheckpointer()
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    # Save the delta pytree. Non-titans weights are None and won't take space
    checkpointer.save(os.path.abspath(output_path), titans_delta)
    print("Surgery complete! You now have the initial Titans weights.")

if __name__ == "__main__":
    # In Colab, you might point this to your Google Drive
    perform_surgery_and_save_delta("./titans_delta_init")
