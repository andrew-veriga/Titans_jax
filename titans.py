import math
from functools import partial
from typing import Callable, Any, Optional, Union, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
# import optax
from einops import rearrange, repeat, pack, unpack

from associative_scan import associative_scan, binary_operator, pad_at_dim


"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
h - heads
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
            
            weight = self.param(f'weight_{ind}', nn.initializers.normal(stddev=0.02), (self.dim, self.dim))
            x = x @ weight
        return x


def default_adaptive_step_transform(adaptive_step, max_lr=1e-2):
    return jax.nn.sigmoid(adaptive_step) * max_lr


def default_loss_fn(pred, target):
    return jnp.mean((pred - target) ** 2, axis=-1)

def init_memory_state(batch_size: int, dim: int, heads: int, dim_head: Optional[int] = None, mlp_depth: int = 2, *, dtype: Any):
    """Standalone function to initialize memory state without Module scope issues."""
    dim_head = default(dim_head, dim // heads if dim_head is None else dim_head)
    
    params = {}
    key = jax.random.PRNGKey(0)
    for i in range(mlp_depth):
        key, subkey = jax.random.split(key)
        params[f'weight_{i}'] = (jax.random.normal(subkey, (dim_head, dim_head)) * 0.02).astype(dtype)
        
    # Expand params to (batch, heads, ...) to be compatible with Gemma's batch dim flattening
    def expand_and_init(p):
        return repeat(p, '... -> b h ...', b=batch_size, h=heads)
        
    initial_weights = jax.tree_util.tree_map(expand_and_init, params)
    momentum = jax.tree_util.tree_map(jnp.zeros_like, initial_weights)
    
    return (initial_weights, momentum)

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
    default_mlp_kwargs: dict = None

    def setup(self):
        dim_head = default(self.dim_head, self.dim // self.heads if self.dim_head is None else self.dim_head)
        dim_inner = dim_head * self.heads

        self.retrieve_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        self.store_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        
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

        self.to_momentum = nn.Dense(self.heads, use_bias=False)
        self.to_adaptive_step = nn.Dense(self.heads, use_bias=False)
        self.to_decay_factor = nn.Dense(self.heads, use_bias=False)

        self.empty_memory_embed = self.param('empty_memory_embed', nn.initializers.normal(stddev=0.02), (self.dim,))

        # prepare per sample grad fn
        def forward_and_loss(params, inputs, loss_weights, target):
            pred = self.memory_model.apply({'params': params}, inputs)
            loss = self.store_memory_loss_fn(pred, target)
            loss = loss * loss_weights
            return loss.sum()

        self.grad_fn = jax.grad(forward_and_loss)

    def init_state(self, batch_size: int, *, dtype: Any):
        mlp_depth = 2
        if exists(self.default_mlp_kwargs):
            mlp_depth = self.default_mlp_kwargs.get('depth', 2)
            
        return init_memory_state(batch_size, self.dim, self.heads, self.dim_head, mlp_depth, dtype=dtype)

    def store_memories(self, seq, past_state):
        batch = seq.shape[0]
        seq = self.store_norm(seq)
        
        seq_len = seq.shape[1]
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)
        seq = seq[:, :round_down_seq_len]

        past_weights, past_momentum = past_state
        
        # adaptive lr, momentum, decay
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = rearrange(adaptive_lr, 'b n h -> b h n')
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        # momentum and decay factor need to be averaged within chunk
        seq_chunked = rearrange(seq, 'b (n c) d -> b n c d', c=self.chunk_size)
        seq_mean = jnp.mean(seq_chunked, axis=2)
        
        adaptive_momentum = jax.nn.sigmoid(self.to_momentum(seq_mean))
        adaptive_momentum = rearrange(adaptive_momentum, 'b n h -> b h n 1')
        
        decay_factor = jax.nn.sigmoid(self.to_decay_factor(seq_mean))
        decay_factor = rearrange(decay_factor, 'b n h -> b h n 1')

        # keys and values
        kv = self.to_keys_values(seq)
        keys, values = jnp.split(kv, 2, axis=-1)

        # multi head
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.heads)
        
        num_chunks = round_down_seq_len // self.chunk_size

        # chunking
        keys = rearrange(keys, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)
        values = rearrange(values, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)
        adaptive_lr_chunked = rearrange(adaptive_lr, 'b h (n c) -> n (b h) c', c=self.chunk_size)

        # Flatten weights over batch and heads to vmap inside scan
        past_weights_bh = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'b h ... -> (b h) ...'),
            past_weights
        )

        def scan_step(carry, xs):
            k, lr, v = xs
            g = jax.vmap(self.grad_fn)(past_weights_bh, k, lr, v)
            return carry, g

        # scan_step_ckp = jax.checkpoint(scan_step)

        _, grads = jax.lax.scan(
            scan_step,
            init=None,
            xs=(keys, adaptive_lr_chunked, values)
        )
        
        # restore to (b h n) ... so subsequent code remains unchanged
        grads = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'n (b h) ... -> (b h n) ...', b=batch, h=self.heads), 
            grads
        )
        
        if exists(self.max_grad_norm):
            grads = jax.tree_util.tree_map(lambda t: softclamp_grad_norm(t, self.max_grad_norm), grads)

        # restore batch, heads and seq dim
        grads = jax.tree_util.tree_map(lambda t: rearrange(t, '(b h n) ... -> b h n ...', b=batch, h=self.heads, n=num_chunks), grads)
        
        # surprises
        surprises = jax.tree_util.tree_map(lambda t: -t, grads)

        # associative scan
        next_momentum = {}
        updates = {}

        for param_name, surprise in surprises.items():
            b, h, n = surprise.shape[:3]
            orig_shape = surprise.shape
            surprise_packed = surprise.reshape(b, h, n, -1)
            
            # associative scan typically operates on axis 1 (sequence)
            # but here it's axis 2. we can rearrange to (b, h, n, ...)
            
            # momentum
            _, momentum = associative_scan(binary_operator, (adaptive_momentum, surprise_packed), axis=2)
            
            # update with decay
            _, update = associative_scan(binary_operator, (1.0 - decay_factor, momentum), axis=2)
            
            updates[param_name] = update.reshape(orig_shape)
            next_momentum[param_name] = momentum.reshape(orig_shape)

        # compute next weights
        last_update = jax.tree_util.tree_map(lambda t: t[:, :, -1], updates)
        next_weights = jax.tree_util.tree_map(lambda p, u: p + u, past_weights, last_update)
        
        last_momentum = jax.tree_util.tree_map(lambda t: t[:, :, -1], next_momentum)
        
        return updates, (next_weights, last_momentum)

    def retrieve_memories(self, seq, weights_with_updates):
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
        
        # 1. Сначала выделяем головы из queries
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.heads)
        
        # 2. Выносим ось чанков (n) вперед
        num_chunks = next_seq_len // self.chunk_size
        queries_chunked = rearrange(queries, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)
        
        # 3. Переворачиваем веса для скана
        weights_chunked = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'b h n ... -> n (b h) ...'), 
            weights_with_updates
        )
        
        def apply_model(p, q):
            return self.memory_model.apply({'params': p}, q)
            
        def retrieve_scan_body(carry, xs):
            w, q = xs
            v = jax.vmap(apply_model)(w, q)
            return carry, v
            
        # retrieve_scan_body_ckp = jax.checkpoint(retrieve_scan_body)
        
        # 4. Идем циклом (scan) по чанкам
        _, values = jax.lax.scan(retrieve_scan_body, None, (weights_chunked, queries_chunked))
        
        # 5. Возвращаем размерности (n (b h) -> b h (n c))
        values = rearrange(values, 'n (b h) c d -> b h (n c) d', b=batch, h=self.heads)
        
        values = self.multihead_rmsnorm(values)
        
        if exists(self.retrieve_gate):
            gate = self.retrieve_gate(seq_curtailed) # (b, n, h)
            gate = rearrange(gate, 'b n h -> b h n 1')
            values = values * gate
            
        values = rearrange(values, 'b h n d -> b n (h d)')
        values = self.combine_heads(values)
        
        # restore, pad with empty memory embed
        empty_embeds = repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=self.chunk_size-1)
        
        values = jnp.concatenate([empty_embeds, values], axis=1)
        
        if padding > 0:
            values = values[:, :-padding]
            
        return values

    def __call__(self, seq, memory_state=None, return_next_memories=False):
        batch, seq_len = seq.shape[:2]
        
        if seq_len < self.chunk_size:
            ret = repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=seq_len)
            if return_next_memories:
                return ret, memory_state
            return ret

        if not exists(memory_state):
            memory_state = self.init_state(batch, dtype=seq.dtype)

        updates, next_mem_state = self.store_memories(seq, memory_state)
        
        past_weights, _ = memory_state
        
        def add_updates(p, u):
            # p: (b, h, ...)
            # u: (b, h, n, ...)
            p_expanded = jnp.expand_dims(p, axis=2)
            return p_expanded + u
            
        weights_with_updates = jax.tree_util.tree_map(add_updates, past_weights, updates)
        
        retrieved = self.retrieve_memories(seq, weights_with_updates)
        
        if not return_next_memories:
            return retrieved
            
        return retrieved, next_mem_state
