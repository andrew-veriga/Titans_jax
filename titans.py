import math
from functools import partial
from typing import Callable, Any, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
import optax
import einx
from einops import rearrange, repeat, pack, unpack

from associative_scan import associative_scan, binary_operator, pad_at_dim


"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
"""


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def round_down_multiple(seq, mult):
    return seq // mult * mult


def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult


def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return (jnp.tanh(t / half_max_value) * half_max_value) + half_max_value


def softclamp_grad_norm(t, max_value):
    # This is trickier in JAX because it's usually applied during gradient computation
    # But we can still have a function that does it on an array.
    t, inverse = pack_one_with_inverse(t, 'bn *')
    
    norm = jnp.linalg.norm(t, axis=-1, keepdims=True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / jnp.maximum(norm, 1e-12))
    return inverse(t)


class MultiheadRMSNorm(nn.Module):
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x):
        # RMSNorm in Flax
        normed = nn.RMSNorm(use_scale=False)(x)
        gamma = self.param('gamma', nn.initializers.zeros, (self.heads, 1, self.dim))
        return normed * (gamma + 1.0)


class MemoryMLP(nn.Module):
    dim: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for ind in range(self.depth):
            is_first = ind == 0
            if not is_first:
                x = nn.silu(x)
            
            weight = self.param(f'weight_{ind}', nn.initializers.normal(), (self.dim, self.dim))
            x = x @ weight
        return x


def default_adaptive_step_transform(adaptive_step, max_lr=1e-2):
    return jax.nn.sigmoid(adaptive_step) * max_lr


def default_loss_fn(pred, target):
    return jnp.mean((pred - target) ** 2, axis=-1)


class NeuralMemory(nn.Module):
    dim: int
    chunk_size: int = 1
    dim_head: Optional[int] = None
    heads: int = 1
    model: Optional[nn.Module] = None
    store_memory_loss_fn: Callable = default_loss_fn
    adaptive_step_transform: Callable = default_adaptive_step_transform
    pre_rmsnorm: bool = True
    post_rmsnorm: bool = True
    max_grad_norm: Optional[float] = None
    # Accelerated scan in JAX is just lax.associative_scan
    default_mlp_kwargs: dict = None

    def setup(self):
        dim_head = default(self.dim_head, self.dim)
        dim_inner = dim_head * self.heads

        self.retrieve_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        self.store_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        
        # post_rmsnorm in original was MultiheadRMSNorm
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, self.heads) if self.post_rmsnorm else identity

        self.combine_heads = nn.Dense(self.dim, use_bias=False) if self.heads > 1 else identity
        
        if self.heads > 1:
            self.retrieve_gate = nn.Sequential([
                nn.Dense(self.heads, use_bias=False),
                nn.sigmoid
            ])
        else:
            self.retrieve_gate = None

        if not exists(self.model):
            mlp_kwargs = default(self.default_mlp_kwargs, {'depth': 2})
            self.memory_model = MemoryMLP(dim_head, **mlp_kwargs)
        else:
            self.memory_model = self.model

        self.to_queries = nn.Dense(dim_inner, use_bias=False)
        self.to_keys_values = nn.Dense(dim_inner * 2, use_bias=False)

        # momentum, adaptive step, decay factor
        self.to_momentum = nn.Dense(self.heads, use_bias=False)
        self.to_adaptive_step = nn.Dense(self.heads, use_bias=False)
        self.to_decay_factor = nn.Dense(self.heads, use_bias=False)

        self.empty_memory_embed = self.param('empty_memory_embed', nn.initializers.normal(stddev=0.02), (self.dim,))

        # prepare per sample grad fn
        def forward_and_loss(params, inputs, loss_weights, target):
            # params here is the pytree of the memory model
            pred = self.memory_model.apply({'params': params}, inputs)
            loss = self.store_memory_loss_fn(pred, target)
            loss = loss * loss_weights
            return loss.sum()

        self.per_sample_grad_fn = jax.vmap(jax.grad(forward_and_loss), in_axes=(None, 0, 0, 0))

    def init_weights_and_momentum(self, rng, dim_head):
        dummy_input = jnp.zeros((1, dim_head))
        variables = self.memory_model.init(rng, dummy_input)
        params = variables['params']
        
        # Momentum has same structure as params
        momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
        return params, momentum

    def store_memories(self, seq, past_state):
        seq = self.store_norm(seq)
        
        seq_len = seq.shape[1]
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)
        seq = seq[:, :round_down_seq_len]

        past_weights, past_momentum = past_state
        
        # adaptive lr, momentum, decay
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = rearrange(adaptive_lr, 'b n h -> (b h) n')
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        # momentum and decay factor need to be averaged within chunk
        seq_chunked = rearrange(seq, 'b (n c) d -> b n c d', c=self.chunk_size)
        seq_mean = jnp.mean(seq_chunked, axis=2)
        
        adaptive_momentum = jax.nn.sigmoid(self.to_momentum(seq_mean))
        adaptive_momentum = rearrange(adaptive_momentum, 'b n h -> (b h) n 1')
        
        decay_factor = jax.nn.sigmoid(self.to_decay_factor(seq_mean))
        decay_factor = rearrange(decay_factor, 'b n h -> (b h) n 1')

        # keys and values
        kv = self.to_keys_values(seq)
        keys, values = jnp.split(kv, 2, axis=-1)

        # multi head
        keys = rearrange(keys, 'b n (h d) -> (b h) n d', h=self.heads)
        values = rearrange(values, 'b n (h d) -> (b h) n d', h=self.heads)
        
        batch_heads = keys.shape[0]

        # chunking
        keys = rearrange(keys, 'b (n c) d -> (b n) c d', c=self.chunk_size)
        values = rearrange(values, 'b (n c) d -> (b n) c d', c=self.chunk_size)
        adaptive_lr_chunked = rearrange(adaptive_lr, 'b (n c) -> (b n) c', c=self.chunk_size)

        # grads
        # We need to apply grads over the batch dimension
        # past_weights is a pytree of params for MemoryMLP
        grads = self.per_sample_grad_fn(past_weights, keys, adaptive_lr_chunked, values)
        
        if exists(self.max_grad_norm):
            grads = jax.tree_util.tree_map(lambda t: softclamp_grad_norm(t, self.max_grad_norm), grads)

        # restore batch and seq dim
        grads = jax.tree_util.tree_map(lambda t: rearrange(t, '(b n) ... -> b n ...', b=batch_heads), grads)
        
        # surprises
        surprises = jax.tree_util.tree_map(lambda t: -t, grads)

        # associative scan
        next_momentum = {}
        updates = {}

        for param_name, surprise in surprises.items():
            batch_heads, n = surprise.shape[:2]
            orig_shape = surprise.shape
            surprise_packed = surprise.reshape(batch_heads, n, -1)
            
            # momentum
            _, momentum = associative_scan(binary_operator, (adaptive_momentum, surprise_packed))
            
            # update with decay
            _, update = associative_scan(binary_operator, (1.0 - decay_factor, momentum))
            
            updates[param_name] = update.reshape(orig_shape)
            next_momentum[param_name] = momentum.reshape(orig_shape)

        # compute next weights
        # last_update: (batch_heads, ...)
        last_update = jax.tree_util.tree_map(lambda t: t[:, -1], updates)
        
        # But wait, past_weights is already (batch_heads, ...) or shared?
        # In the original, past_weights was (heads, ...)? No, it was TensorDict.
        # Actually, in PyTorch it was shared across heads? 
        # "past_weights, past_momentum = past_state"
        # "curr_weights = curr_weights + past_weights"
        
        # We need to be careful with shapes here.
        # In PyTorch, weights were (dim, dim) and past_weights were added to them.
        # If past_weights are (batch, heads, ...), we need to handle that.
        
        next_weights = jax.tree_util.tree_map(lambda p, u: p + u, past_weights, last_update)
        
        return updates, (next_weights, next_momentum)

    def retrieve_memories(self, seq, past_weights=None):
        batch, seq_len = seq.shape[:2]
        seq = self.retrieve_norm(seq)
        
        # truncate
        seq_curtailed = seq[:, (self.chunk_size - 1):]
        curtailed_seq_len = seq_curtailed.shape[1]
        
        next_seq_len = round_up_multiple(curtailed_seq_len, self.chunk_size)
        padding = next_seq_len - curtailed_seq_len
        
        if padding > 0:
            seq_curtailed = pad_at_dim(seq_curtailed, (0, padding), dim=1)
        
        queries = self.to_queries(seq_curtailed)
        queries = rearrange(queries, 'b n (h d) -> (b h) n d', h=self.heads)
        
        # fetch from memory model
        # queries: (batch*heads, next_seq_len, dim_head)
        # past_weights: pytree where each leaf is (batch*heads, next_seq_len, ...) or (batch*heads, ...)
        
        # Actually, in original retrieve_memories:
        # curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        # queries = rearrange(queries, 'b (n c) d -> (b n) c d', c = chunk_size)
        # values = functional_call(self.memory_model, dict(curr_weights), queries)
        
        # We need to vmap the memory model apply over (batch*heads * n)
        batch_heads = queries.shape[0]
        queries_chunked = rearrange(queries, 'b (n c) d -> (b n) c d', c=self.chunk_size)
        
        # past_weights here are actually (batch_heads, n, ...) from updates
        # Wait, the past_weights + updates in forward was:
        # retrieved = self.retrieve_memories(seq, past_weights + updates)
        # updates are (batch_heads, n, ...)
        
        # So we need to apply the model for each chunk with its corresponding weight.
        
        def apply_model(p, q):
            return self.memory_model.apply({'params': p}, q)
        
        # past_weights_chunked: (batch_heads * n, ...)
        # past_weights: (batch_heads, n, ...)
        past_weights_chunked = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'b n ... -> (b n) ...'), 
            past_weights
        )
        
        values = jax.vmap(apply_model)(past_weights_chunked, queries_chunked)
        
        # restore batch dim
        values = rearrange(values, '(b h n) c d -> b h (n c) d', b=batch, h=self.heads)
        
        values = self.multihead_rmsnorm(values)
        
        if exists(self.retrieve_gate):
            # seq_curtailed: (b, n, d)
            gate = self.retrieve_gate(seq_curtailed) # (b, n, h)
            gate = rearrange(gate, 'b n h -> b h n 1')
            values = values * gate
            
        values = rearrange(values, 'b h n d -> b n (h d)')
        values = self.combine_heads(values)
        
        # restore, pad with empty memory embed
        # The original had learned empty memory embed.
        empty_embeds = repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=self.chunk_size-1)
        
        values = jnp.concatenate([empty_embeds, values], axis=1)
        
        if padding > 0:
            values = values[:, :-padding]
            
        return values

    def __call__(self, seq, store_seq=None, past_state=None, return_next_memories=False):
        batch, seq_len = seq.shape[:2]
        
        if seq_len < self.chunk_size:
            return repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=seq_len)

        if not exists(past_state):
            # In Flax, we usually don't initialize state inside __call__ like this
            # but for a direct refactoring we can handle it or assume it's passed.
            # To be more Flax-idiomatic, we should probably handle this in a separate state.
            # However, I'll follow the original structure as much as possible.
            dim_head = default(self.dim_head, self.dim)
            rng = self.make_rng('params')
            past_state = self.init_weights_and_momentum(rng, dim_head)

        store_seq = default(store_seq, seq)
        
        updates, next_memories = self.store_memories(store_seq, past_state)
        
        past_weights, _ = past_state
        # Add updates to past weights
        # past_weights: (heads, ...)? No, should be (batch_heads, ...)
        # We need to broadcast past_weights to (batch_heads, n, ...) then add updates
        
        def add_updates(p, u):
            # p: (...) or (batch_heads, ...)
            # u: (batch_heads, n, ...)
            if p.ndim == u.ndim - 1:
                p_expanded = jnp.expand_dims(p, axis=1)
            else:
                p_expanded = jnp.expand_dims(p, axis=(0, 1))
            return p_expanded + u
            
        weights_with_updates = jax.tree_util.tree_map(add_updates, past_weights, updates)
        
        retrieved = self.retrieve_memories(seq, weights_with_updates)
        
        if not return_next_memories:
            return retrieved
            
        return retrieved, next_memories
