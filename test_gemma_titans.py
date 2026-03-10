import jax
import jax.numpy as jnp
import flax.linen as nn
from gemma.gm.nn import _config, _modules
from gemma_titans import GemmaTitansTransformer
import time

def test_hybrid_model():
    # 1. Create a small Gemma config for testing
    config = _config.TransformerConfig(
        num_embed=1000,
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        head_dim=32,
        num_kv_heads=1,
        final_logit_softcap=None,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        attention_types=[_modules.AttentionType.GLOBAL] * 2,
    )
    
    # 2. Instantiate the hybrid transformer
    model = GemmaTitansTransformer(config=config, dtype=jnp.float32)
    
    # 3. Initialize parameters and cache
    batch_size = 2
    seq_len = 16
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print("Initializing model...")
    rng = jax.random.PRNGKey(0)
    init_rng, task_rng = jax.random.split(rng)
    
    variables = model.init(init_rng, tokens=tokens)
    params = variables['params']
    
    print("Initializing cache...")
    cache = model.init_cache(batch_size=batch_size, dtype=jnp.float32, cache_length=seq_len)
    
    # 4. Forward pass
    print("Running forward pass...")
    start = time.time()
    output = model.apply({'params': params}, tokens=tokens, cache=cache)
    
    # The output is a struct with 'logits' and 'cache'
    logits = output.logits
    new_cache = output.cache
    
    end = time.time()
    print(f"Forward pass completed in {end - start:.4f}s")
    print(f"Logits shape: {logits.shape}")
    
    # Check if cache contains memory_state
    for layer_name, layer_cache in new_cache.items():
        print(f"Layer {layer_name} cache keys: {layer_cache.__class__.__name__}")
        if hasattr(layer_cache, 'memory_state'):
            print(f"  Memory state found in {layer_name}")

    print("Test passed!")

if __name__ == "__main__":
    test_hybrid_model()
