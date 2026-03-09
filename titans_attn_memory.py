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


class MemoryAttention(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        assert x.shape[-2] > 1, 'chunk size needs to be greater than 1 for using attention as memory'

        wq = self.param('wq', nn.initializers.normal(), (self.dim, self.dim))
        wk = self.param('wk', nn.initializers.normal(), (self.dim, self.dim))
        wv1 = self.param('wv1', nn.initializers.normal(), (self.dim, self.dim))
        wv2 = self.param('wv2', nn.initializers.normal(), (self.dim, self.dim))

        q = x @ wq
        k = x @ wk
        v = x @ wv1

        # Manual causal attention
        dim_head = q.shape[-1]
        dots = (q @ k.T) * (dim_head ** -0.5)
        
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        dots = jnp.where(mask, dots, -1e9)
        
        attn = jax.nn.softmax(dots, axis=-1)
        hidden = attn @ v

        return nn.silu(hidden) @ wv2


def default_loss_fn(pred, target):
    return jnp.sum(jnp.mean((pred - target) ** 2, axis=-1))


class NeuralMemory(nn.Module):
    dim: int
    chunk_size: int = 1
    dim_head: Optional[int] = None
    heads: int = 1
    model: Optional[nn.Module] = None
    store_memory_loss_fn: Callable = default_loss_fn
    pre_rmsnorm: bool = True
    post_rmsnorm: bool = True
    # Accelerated scan in JAX is just lax.associative_scan
    default_model_kwargs: dict = None

    def setup(self):
        dim_head = default(self.dim_head, self.dim)
        dim_inner = dim_head * self.heads

        self.retrieve_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else lambda x: x
        self.store_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else lambda x: x
        self.post_rmsnorm_layer = nn.RMSNorm(use_scale=True) if self.post_rmsnorm else lambda x: x

        self.combine_heads = nn.Dense(self.dim, use_bias=False) if self.heads > 1 else lambda x: x

        if not exists(self.model):
            model_kwargs = default(self.default_model_kwargs, {})
            self.memory_model = MemoryAttention(dim_head, **model_kwargs)
        else:
            self.memory_model = self.model

        self.to_queries = nn.Dense(dim_inner, use_bias=False)
        self.to_keys_values = nn.Dense(dim_inner * 2, use_bias=False)

        # momentum, adaptive step, decay factor
        self.to_momentum = nn.Dense(self.heads, use_bias=False)
        self.to_adaptive_step = nn.Dense(self.heads, use_bias=False)
        self.to_decay_factor = nn.Dense(self.heads, use_bias=False)

        def forward_and_loss(params, inputs, target):
            pred = self.memory_model.apply({'params': params}, inputs)
            loss = self.store_memory_loss_fn(pred, target)
            return loss

        self.per_sample_grad_fn = jax.vmap(jax.grad(forward_and_loss), in_axes=(None, 0, 0))

    def init_weights_and_momentum(self, rng, dim_head):
        # We need a dummy input of size (chunk_size, dim_head)
        # Assuming chunk_size > 1 for attention memory
        dummy_input = jnp.zeros((self.chunk_size, dim_head))
        variables = self.memory_model.init(rng, dummy_input)
        params = variables['params']
        momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
        return params, momentum

    def store_memories(self, seq, past_state):
        seq = self.store_norm(seq)
        
        seq_len = seq.shape[1]
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)
        seq = seq[:, :round_down_seq_len]

        past_weights, past_momentum = past_state
        
        # adaptive lr, momentum, decay
        adaptive_step = self.to_adaptive_step(seq) # (b, n, h)
        adaptive_lr = jnp.exp(jax.nn.sigmoid(adaptive_step) * -15.0)
        adaptive_lr = rearrange(adaptive_lr, 'b n h -> (b h) n')
        adaptive_lr = rearrange(adaptive_lr, 'b (n c) -> b n c', c=self.chunk_size).mean(axis=-1)

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

        # grads
        grads = self.per_sample_grad_fn(past_weights, keys, values)
        
        # restore batch and seq dim
        grads = jax.tree_util.tree_map(lambda t: rearrange(t, '(b n) ... -> b n ...', b=batch_heads), grads)
        
        # surprises
        # apply adaptive_lr: einx.multiply('b n ..., b n -> b n ...', t, -adaptive_lr)
        def apply_lr(g, lr):
            # g: (batch_heads, n, ...)
            # lr: (batch_heads, n)
            lr_expanded = lr.reshape(lr.shape + (1,) * (g.ndim - 2))
            return g * (-lr_expanded)
            
        surprises = jax.tree_util.tree_map(lambda t: apply_lr(t, adaptive_lr), grads)

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
        last_update = jax.tree_util.tree_map(lambda t: t[:, -1], updates)
        next_weights = jax.tree_util.tree_map(lambda p, u: p + u, past_weights, last_update)
        
        return updates, (next_weights, next_momentum)

    def retrieve_memories(self, seq, past_weights=None):
        batch, seq_len = seq.shape[:2]
        seq = self.retrieve_norm(seq)
        
        assert seq_len > self.chunk_size
        
        seq_curtailed = seq[:, self.chunk_size:]
        curtailed_seq_len = seq_curtailed.shape[1]
        
        next_seq_len = round_up_multiple(curtailed_seq_len + 1, self.chunk_size)
        padding = next_seq_len - curtailed_seq_len
        
        seq_curtailed = pad_at_dim(seq_curtailed, (0, padding), dim=1)
        
        queries = self.to_queries(seq_curtailed)
        queries = rearrange(queries, 'b n (h d) -> (b h) n d', h=self.heads)
        
        batch_heads = queries.shape[0]
        queries_chunked = rearrange(queries, 'b (n c) d -> (b n) c d', c=self.chunk_size)
        
        def apply_model(p, q):
            return self.memory_model.apply({'params': p}, q)
        
        past_weights_chunked = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'b n ... -> (b n) ...'), 
            past_weights
        )
        
        values = jax.vmap(apply_model)(past_weights_chunked, queries_chunked)
        
        # restore batch dim
        values = rearrange(values, '(b n) c d -> b (n c) d', b=batch_heads)
        
        # merge heads and combine
        values = rearrange(values, '(b h) n d -> b n (h d)', h=self.heads)
        values = self.combine_heads(values)
        
        values = self.post_rmsnorm_layer(values)
        
        # restore padding
        values = pad_at_dim(values, (self.chunk_size, 0), dim=1, value=0.0)
        values = values[:, :-padding]
            
        return values

    def __call__(self, seq, store_seq=None, past_state=None, return_next_memories=False):
        batch, seq_len = seq.shape[:2]
        
        if seq_len <= self.chunk_size:
            return jnp.zeros_like(seq)

        if not exists(past_state):
            dim_head = default(self.dim_head, self.dim)
            rng = self.make_rng('params')
            past_state = self.init_weights_and_momentum(rng, dim_head)

        store_seq = default(store_seq, seq)
        
        updates, next_memories = self.store_memories(store_seq, past_state)
        
        past_weights, _ = past_state
        
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
