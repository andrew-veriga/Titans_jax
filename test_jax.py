import jax
import jax.numpy as jnp
from titans import NeuralMemory as MLPNeuralMemory
from titans_attn_memory import NeuralMemory as AttnNeuralMemory

def test_mlp_memory():
    print("Testing MLP Neural Memory...")
    dim = 64
    heads = 4
    chunk_size = 8
    seq_len = 32
    batch = 2
    
    model = MLPNeuralMemory(dim=dim, heads=heads, chunk_size=chunk_size)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch, seq_len, dim))
    
    variables = model.init(rng, x)
    out = model.apply(variables, x, rngs={'params': rng})
    print(f"MLP Output shape: {out.shape}")
    assert out.shape == (batch, seq_len, dim)

def test_attn_memory():
    print("Testing Attention Neural Memory...")
    dim = 64
    heads = 4
    chunk_size = 8
    seq_len = 32
    batch = 2
    
    model = AttnNeuralMemory(dim=dim, heads=heads, chunk_size=chunk_size)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch, seq_len, dim))
    
    variables = model.init(rng, x)
    out = model.apply(variables, x, rngs={'params': rng})
    print(f"Attn Output shape: {out.shape}")
    assert out.shape == (batch, seq_len, dim)

if __name__ == "__main__":
    try:
        test_mlp_memory()
        test_attn_memory()
        print("All tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
