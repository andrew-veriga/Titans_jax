import math
from functools import partial
from typing import Callable, Any, Optional, Union, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import flax.linen as nn
import optax
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


def default_loss_fn(pred, target, **kwargs):
    return jnp.mean((pred - target) ** 2, axis=-1)

# Huber (delta подбирается, старт с 0.1):
def huber_loss(pred, target, delta=0.1, **kwargs):
    r = pred - target
    loss = jnp.where(
        jnp.abs(r) <= delta,
        0.5 * r ** 2,
        delta * (jnp.abs(r) - 0.5 * delta)
    )
    return jnp.mean(loss, axis=-1)

def init_memory_state(batch_size: int, dim: int, neural_mem_kwargs: dict, *, dtype):
    mem = default(neural_mem_kwargs, {})
    heads = mem.get('heads', 1)
    dim_head = mem.get('dim_head', dim // heads)
    mlp_depth = mem.get('mlp_depth', 2)
    
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

def newton_schulz_norm_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """
    Спектральная нормализация матрицы через итерации Ньютона-Шульца (Newton-Schulz).
    Оптимизировано для JAX/TPU. Нормализует одну 2D-матрицу.

    Args:
        x: Входная матрица апдейтов (например, сюрприз памяти).
        steps: Количество итераций (обычно 5 достаточно для сходимости).
        eps: Эпсилон для числовой стабильности.
    Returns:
        Спектрально нормализованная матрица той же размерности.
    """
    steps = 5
    eps = 1e-7
    # Для NS требуется квадратная или широкая матрица. 
    # Если строк больше чем столбцов, транспонируем временно.
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True

    a, b, c = (1.5, -0.5, 5) # Константы из реализации Muon/Titans
    
    # Шаг 1: Масштабирование (делим на норму Фробениуса)
    norm = jnp.linalg.norm(x, ord='fro')
    x_scaled = x / (norm + eps)

    # Шаг 2: Итерации Ньютона-Шульца
    def ns_step(i, x_curr):
        return a * x_curr + b * (x_curr @ x_curr.T @ x_curr)
        
    x_normalized = jax.lax.fori_loop(0, steps, ns_step, x_scaled)

    # Возвращаем исходную ориентацию, если транспонировали
    if transposed:
        x_normalized = x_normalized.T
        
    return x_normalized
import jax.numpy as jnp

def apply_fast_ns_to_tensor(t: jnp.ndarray) -> jnp.ndarray:
    """
    Векторизованная и развернутая (unrolled) версия Newton-Schulz.
    Работает мгновенно благодаря нативному батчингу jnp.matmul и слиянию графа XLA.
    """
    """
    Сверхбыстрая версия Newton-Schulz для TPU/GPU.
    Использует сплющивание в 3D, вычисления в bfloat16 и развернутый цикл.
    """
    steps= 3
    eps = 1e-7
    if t.ndim < 5:
        return t
        
    orig_shape = t.shape
    d1, d2 = orig_shape[-2], orig_shape[-1]
    t_3d = t.reshape(-1, d1, d2)
    
    should_transpose = d1 > d2
    if should_transpose:
        t_3d = jnp.swapaxes(t_3d, -1, -2)
        
    orig_dtype = t_3d.dtype
    t_3d = t_3d.astype(jnp.bfloat16)
    
    norm = jnp.linalg.norm(t_3d, ord='fro', axis=(-2, -1), keepdims=True)
    t_3d = t_3d / jnp.maximum(norm, eps)
    
    # "Агрессивные" коэффициенты из Titans-PyTorch
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for _ in range(steps):
        A = t_3d @ jnp.swapaxes(t_3d, -1, -2)
        B = b * A + c * (A @ A)
        t_3d = a * t_3d + B @ t_3d
        
    if should_transpose:
        t_3d = jnp.swapaxes(t_3d, -1, -2)
        
    t_3d = t_3d.astype(orig_dtype)
    return t_3d.reshape(orig_shape)

def apply_ns_to_tensor(t: jnp.ndarray) -> jnp.ndarray:
    """
    Применяет Newton-Schulz normalization ко всем матрицам в тензоре формы (b, h, n, ...).
    Если это bias (1D вектор на уровне параметров), просто возвращаем его или делаем L2-нормализацию.
    """
    # Если параметр — вектор (например, bias формы (b, h, n, d))
    if t.ndim <= 4: 
        # Можно оставить как есть, либо сделать L2 нормализацию
        return t 
            
    # Если параметр — матрица (форма (b, h, n, d1, d2))
    elif t.ndim == 5:
        # Трижды применяем vmap, чтобы "провалиться" сквозь batch, heads и chunks
        # и применить NS непосредственно к матрице (d1, d2)
        vmap_ns = jax.vmap(jax.vmap(jax.vmap(newton_schulz_norm_matrix, in_axes=0), in_axes=0), in_axes=0)
        return vmap_ns(t)
    
    return t

class NeuralMemory(nn.Module):
    dim: int
    neural_mem_kwargs: dict = None  # <-- единый словарь
    # chunk_size: int = 1
    # dim_head: Optional[int] = None
    # heads: int = 1
    # store_memory_loss_fn: Callable = default_loss_fn

    # max_grad_norm: Optional[float] = None
    # elastic_net_lambda: Optional[float] = None
    # default_mlp_kwargs: dict = None
    # diff_view: bool = False
    # is_look_ahead: bool = False

    def setup(self):
        self.adaptive_step_transform = default_adaptive_step_transform
        self.pre_rmsnorm = True
        self.post_rmsnorm = True
        
        mem = default(self.neural_mem_kwargs, {})
        self.heads = mem.get('heads', 1)
        self.dim_head = mem.get('dim_head', self.dim // self.heads)
        self.chunk_size = mem.get('chunk_size', 1)
        self.max_grad_norm = mem.get('max_grad_norm', None)
        self.elastic_net_lambda = mem.get('elastic_net_lambda', None)
        self.mlp_depth = mem.get('mlp_depth', 2)
        self.diff_view = mem.get('diff_view', False)
        self.is_look_ahead = mem.get('is_look_ahead', False)
        self.store_memory_loss_fn = mem.get('store_memory_loss_fn', default_loss_fn)

        

        self.retrieve_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        self.store_norm = nn.RMSNorm(use_scale=True) if self.pre_rmsnorm else identity
        
        self.multihead_rmsnorm = MultiheadRMSNorm(self.dim_head, self.heads) if self.post_rmsnorm else identity

        self.combine_heads = nn.Dense(self.dim, use_bias=False) if self.heads > 1 else identity
        
        if self.heads > 1:
            self.retrieve_gate = nn.Sequential([
                nn.Dense(self.heads, use_bias=False),
                nn.sigmoid
            ])
        else:
            self.retrieve_gate = None

        self.memory_model = MemoryMLP(self.dim_head, depth=self.mlp_depth)

        dim_inner = self.dim_head * self.heads
        self.to_queries = nn.Dense(dim_inner, use_bias=False)
        self.to_keys = nn.Dense(dim_inner, use_bias=False)
        self.to_keys_values = nn.Dense(dim_inner * 2, use_bias=False)

        self.to_momentum = nn.Dense(self.heads, use_bias=False)

        self.to_adaptive_step = nn.Dense(self.heads * self.mlp_depth, use_bias=False)
        self.to_decay_factor = nn.Dense(self.heads, use_bias=False)

        self.empty_memory_embed = self.param('empty_memory_embed', nn.initializers.normal(stddev=0.02), (self.dim,))

        # prepare per sample grad fn
        def forward_and_loss(params, inputs, target, **loss_kwargs):
            pred = self.memory_model.apply({'params': params}, inputs)
            loss = self.store_memory_loss_fn(pred, target, **loss_kwargs)
            return loss.sum()

        self.grad_fn = forward_and_loss

        # attention pooling для получения весов внутри чанков для momentum и decay factor:
        # Объявляем слои для attention pooling как атрибуты модуля
        pool_hidden_dim = self.dim // self.heads
        self.chunk_pool_layer1 = nn.Dense(pool_hidden_dim, use_bias=False)
        self.chunk_pool_layer2 = nn.Dense(1, use_bias=False)


    def init_state(self, batch_size: int, *, dtype: Any):
        return init_memory_state(batch_size, self.dim, self.neural_mem_kwargs, dtype=dtype)

    def store_memories(self, seq, past_state, kv_seq=None, loss_kwargs=None):
        """
        реализует механизм ассоциативной памяти, 
        где веса модели обновляются на лету на основе входной последовательности
        """
        loss_kwargs = default(loss_kwargs, {})
        batch = seq.shape[0]
        seq = self.store_norm(seq)
        
        kv_seq = default(kv_seq, seq)
        if self.diff_view:
            kv_seq = self.store_norm(kv_seq)

        seq_len = seq.shape[1]
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)
        seq = seq[:, :round_down_seq_len]
        kv_seq = kv_seq[:, :round_down_seq_len]

        past_weights, past_momentum = past_state
        
        # adaptive lr, momentum, decay
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = rearrange(adaptive_lr, 'b n (h d) -> b h n d', h=self.heads)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        # momentum and decay factor need to be averaged within chunk
        seq_chunked = rearrange(seq, 'b (n c) d -> b n c d', c=self.chunk_size)
        
        # seq_mean = jnp.mean(seq_chunked, axis=2)

        ##### attention pooling #########
        # 1. Пропускаем токены через слои
        hidden = self.chunk_pool_layer1(seq_chunked)
        hidden = nn.silu(hidden)
        attn_logits = self.chunk_pool_layer2(hidden)

        # 2. Нормализуем их внутри чанка (softmax вдоль оси токенов 'c')
        attn_weights = jax.nn.softmax(attn_logits, axis=2) 

        # 3. Взвешенная сумма вместо простого среднего
        seq_mean = jnp.sum(seq_chunked * attn_weights, axis=2)
        ###################################################

        adaptive_momentum = jax.nn.sigmoid(self.to_momentum(seq_mean))
        adaptive_momentum = rearrange(adaptive_momentum, 'b n h -> b h n 1')
        
        decay_factor = jax.nn.sigmoid(self.to_decay_factor(seq_mean))
        decay_factor = rearrange(decay_factor, 'b n h -> b h n 1')

        # keys and values (extracted from kv_seq if diff_view is True)
        kv = self.to_keys_values(kv_seq)
        keys, values = jnp.split(kv, 2, axis=-1)

        # multi head
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.heads)
        
        if self.is_look_ahead:
            # --- ВНЕДРЕНИЕ VALUE LOOKAHEAD ---
            # Сдвигаем значения на 1 токен в будущее вдоль оси seq_len (ось 2)
            values_next = values[:, :, 1:, :]
            # Паддим последний токен нулями (или можно задублировать последний)
            zeros_pad = jnp.zeros_like(values[:, :, -1:, :])
            values_lookahead = jnp.concatenate([values_next, zeros_pad], axis=2)
            
            # Крайне важно: stop_gradient! Мы хотим обновить только память, 
            # а не менять способ генерации эталонного value базовой моделью.
            values = jax.lax.stop_gradient(values_lookahead)
            # ---------------------------------
        else:
            values = jax.lax.stop_gradient(values)

        num_chunks = round_down_seq_len // self.chunk_size

        # chunking
        keys = rearrange(keys, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)
        values = rearrange(values, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)
        adaptive_lr_chunked = rearrange(adaptive_lr, 'b h (n c) d -> n (b h) c d', c=self.chunk_size)

        # Flatten weights over batch and heads to vmap inside scan
        past_weights_bh = jax.tree_util.tree_map(
            lambda t: rearrange(t, 'b h ... -> (b h) ...'),
            past_weights
        )
        
        # Create the gradient function from forward_and_loss
        grad_fn = jax.grad(self.grad_fn)

        def scan_step(carry, xs):
            k, lr, v = xs
            # lr сейчас имеет форму (batch*heads, chunk_size, depth)

            # Guard 1: clip keys/values before grad_fn to prevent gradient overflow
            k = jnp.clip(k, -10.0, 10.0)
            v = jnp.clip(v, -10.0, 10.0)

            # 1. Получаем сырые градиенты (PyTree словарей с 'weight_0', 'weight_1' и т.д.)
            # Заметь: grad_fn больше не принимает lr
            g = jax.vmap(partial(grad_fn, **loss_kwargs))(past_weights_bh, k, v)

            # Guard 2: zero out any NaN/Inf gradients before applying them
            g = jax.tree_util.tree_map(
                lambda t: jnp.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), g
            )
            
            # 2. Модуляция: умножаем градиенты каждого слоя на его собственный lr.
            # Для этого усредняем lr внутри чанка по оси 1 (chunk_size), 
            # чтобы получить один LR для батча/головы на этот чанк (форма: batch*heads, depth)
            lr_mean = jnp.mean(lr, axis=1)   
            
            # Применяем: expanding dims, чтобы скаляр (batch*heads, 1, 1) умножился на матрицу (batch*heads, dim, dim)
            for i in range(self.mlp_depth):
                g[f'weight_{i}'] = g[f'weight_{i}'] * lr_mean[:, i][:, None, None]

            return carry, g

        scan_step_ckp = jax.checkpoint(scan_step)

        _, grads = jax.lax.scan(
            scan_step_ckp,
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
        
        # Сначала получаем "сырые" сюрпризы (инвертированные градиенты)
        surprises = jax.tree_util.tree_map(lambda t: -t, grads)

        # Применяем спектральную нормализацию Ньютона-Шульца ко всем матрицам (нет)
        # surprises = jax.tree_util.tree_map(apply_fast_ns_to_tensor, surprises)


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
            # Elastic Net soft thresholding:
            if exists(self.elastic_net_lambda):
                update = jnp.sign(update) * jnp.maximum(jnp.abs(update) - self.elastic_net_lambda, 0.0)

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

    def __call__(self, seq, memory_state=None, return_next_memories=False, kv_seq=None, loss_kwargs=None):
        batch, seq_len = seq.shape[:2]
        
        if seq_len < self.chunk_size:
            ret = repeat(self.empty_memory_embed, 'd -> b n d', b=batch, n=seq_len)
            if return_next_memories:
                return ret, memory_state
            return ret

        if not exists(memory_state):
            memory_state = self.init_state(batch, dtype=seq.dtype)

        updates, next_mem_state = self.store_memories(seq, memory_state, kv_seq=kv_seq, loss_kwargs=loss_kwargs)
        
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
