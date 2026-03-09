import math
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad
import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from tensordict import TensorDict

from associative_scan import associative_scan, binary_operator, pad_at_dim


"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
"""


# Use partial to create a linear layer version without bias
LinearNoBias = partial(Linear, bias = False)


def exists(v):
    """
    Check if a variable exists (is not None).

    Args:
        v (Any): Any variable.

    Returns:
        bool: True if v is not None, False otherwise.
    """
    return v is not None


def default(v, d):
    """
    Returns the variable itself if it exists (is not None); otherwise, returns a default value.

    Args:
        v (Any): Any variable.
        d (Any): Default value.

    Returns:
        Any: v if it exists; otherwise, d.
    """
    return v if exists(v) else d


def identity(t):
    """
    Returns the input tensor as is.

    Args:
        t (Tensor): Input tensor.

    Returns:
        Tensor: Input tensor.
    """
    return t


def round_down_multiple(seq, mult):
    """
    Rounds down the sequence length to the nearest multiple of a specified value.

    Args:
        seq (int): Sequence length.
        mult (int): Multiple.

    Returns:
        int: Rounded-down sequence length.
    """
    return seq // mult * mult


def round_up_multiple(seq, mult):
    """
    Rounds up the sequence length to the nearest multiple of a specified value.

    Args:
        seq (int): Sequence length.
        mult (int): Multiple.

    Returns:
        int: Rounded-up sequence length.
    """
    return math.ceil(seq / mult) * mult


def pack_one_with_inverse(t, pattern):
    """
    Packs a tensor and returns an inverse function for unpacking.

    Args:
        t (Tensor): Tensor to pack.
        pattern (Tuple[int, ...]): Packing pattern.

    Returns:
        Tuple[Tensor, Callable]: Packed tensor and an inverse function for unpacking.
    """
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        """
        Unpacks a tensor.

        Args:
            out (Tensor): Tensor to unpack.
            inv_pattern (Tuple[int, ...], optional): Unpacking pattern, defaults to None.

        Returns:
            Tensor: Unpacked tensor.
        """
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


def softclamp_max(t, max_value):
    """
    Applies soft clamping to a tensor to limit its maximum value.

    Args:
        t (Tensor): Input tensor.
        max_value (float): Maximum value.

    Returns:
        Tensor: Soft-clamped tensor.
    """
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value


def softclamp_grad_norm(t, max_value):
    """
    Applies soft clamping to gradient norms.

    Args:
        t (Tensor): Input tensor.
        max_value (float): Maximum norm.

    Returns:
        Tensor: Tensor with soft-clamped gradient norm.
    """
    # Pack tensor to restore original shape on unpacking
    t, inverse = pack_one_with_inverse(t, 'bn *')
    
    # Calculate gradient norm
    norm = t.norm(dim = -1, keepdim = True)
    # Soft clamp the norm
    clamped_norm = softclamp_max(norm, max_value)

    # Adjust gradient based on norm ratio
    t = t * (clamped_norm / norm)
    # Unpack tensor to restore original shape
    return inverse(t)


class MultiheadRMSNorm(Module):
    """
    Multihead RMSNorm module.

    Applies RMS normalization and scales each head using multi-head parameters.
    """
    def __init__(self, dim, heads):
        """
        Initializes the Multihead RMSNorm module.

        Args:
            dim (int): Feature dimension.
            heads (int): Number of heads.
        """
        super().__init__()
        # Initialize RMSNorm layer without learnable affine parameters
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        # Initialize multi-head scale parameters, shape (heads, 1, dim)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor, shape (batch_size, ..., dim).

        Returns:
            Tensor: Normalized and scaled tensor, same shape as input.
        """
        # Apply RMS normalization to input tensor
        # Scale the normalized tensor using multi-head scale parameters
        # gamma shape (heads, 1, dim) aligns with normed via broadcasting
        return self.rmsnorm(x) * (self.gamma + 1.)


class MemoryMLP(Module):
    """
    Memory MLP module.

    Consists of multiple linear layers, each followed by a SiLU activation (except the first linear layer).
    """
    def __init__(
        self,
        dim,
        depth
    ):
        """
        Initializes the Memory MLP module.

        Args:
            dim (int): Input and output feature dimension.
            depth (int): MLP depth, i.e., number of linear layers.
        """
        super().__init__()
        # Initialize parameter list, each parameter is a weight matrix for a linear layer, shape (dim, dim)
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(depth)])

    def forward(
        self,
        x
    ):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor, shape (batch_size, ..., dim).

        Returns:
            Tensor: MLP output, same shape as input.
        """
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                # Apply SiLU activation if not the first linear layer
                x = F.silu(x)

            # Apply linear layer
            x = x @ weight

        return x


def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    """
    Default adaptive step transform function.

    Converts adaptive step to learning rate, ranging from 0 to max_lr.

    Args:
        adaptive_step (Tensor): Adaptive step tensor.
        max_lr (float, optional): Maximum learning rate, defaults to 1e-2.

    Returns:
        Tensor: Transformed learning rate tensor.
    """
    return adaptive_step.sigmoid() * max_lr


def default_loss_fn(pred, target):
    """
    Default loss function.

    Calculates Mean Squared Error (MSE) between prediction and target.

    Args:
        pred (Tensor): Predicted values.
        target (Tensor): Target values.

    Returns:
        Tensor: Computed loss.
    """
    return (pred - target).pow(2).mean(dim = -1)


class NeuralMemory(Module):
    """
    Neural Memory Module.

    Implements a neural memory mechanism that stores and retrieves information via a memory model, 
    dynamically adjusting learning rate and momentum during training.
    """
    def __init__(
        self,
        dim,
        chunk_size = 1,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn = default_loss_fn,
        adaptive_step_transform = default_adaptive_step_transform,
        pre_rmsnorm = True,
        post_rmsnorm = True,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        default_mlp_kwargs: dict = dict(
            depth = 2
        )
    ):
        """
        Initializes the Neural Memory Module.

        Args:
            dim (int): Feature dimension.
            chunk_size (int, optional): Chunk size, defaults to 1.
            dim_head (int, optional): Dimension of each attention head, defaults to dim if None.
            heads (int, optional): Number of attention heads, defaults to 1.
            model (Module, optional): Memory model, defaults to None. If None, default `MemoryMLP` model is used.
            store_memory_loss_fn (Callable[[Tensor, Tensor], Tensor], optional): Loss function for storing memory, defaults to MSE.
            adaptive_step_transform (Callable[[Tensor], Tensor], optional): Adaptive step transform function.
            pre_rmsnorm (bool, optional): Whether to apply RMSNorm before storage, defaults to True.
            post_rmsnorm (bool, optional): Whether to apply RMSNorm after storage, defaults to True.
            max_grad_norm (float, optional): Maximum gradient norm for storing memories, defaults to None.
            use_accelerated_scan (bool, optional): Whether to use accelerated scan, defaults to False.
            default_mlp_kwargs (Dict[str, Any], optional): Default MLP parameters, defaults to depth 2.
        """
        super().__init__()
        dim_head = default(dim_head, dim)

        # norms
        # Normalization layers
        # RMSNorm before retrieval
        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # RMSNorm before storage
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # Multi-head RMSNorm after storage
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        # maybe multi-headed
        # Multi-head processing
        # Calculate inner feature dimension
        dim_inner = dim_head * heads

        self.heads = heads

        # Merge batch and head dimensions
        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)
        # Separate head and batch dimensions
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        # If multiple heads, use linear layer to merge; otherwise, use identity
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = nn.Sequential(
            LinearNoBias(dim, heads),  # linear layer mapping feature dimension to number of heads
            Rearrange('b n h -> b h n 1'),  # reshape tensor
            nn.Sigmoid()  # apply Sigmoid activation
        ) if heads > 1 else None  # no gating mechanism if only one head

        # memory mlp
        # Memory model
        if not exists(model):
            # Use default `MemoryMLP` if no memory model provided
            model = MemoryMLP(dim_head, **default_mlp_kwargs)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # the memory is the weights of the model
        self.memory_model = model

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared
        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func
        def forward_and_loss(params, inputs, loss_weights, target):
            # Forward pass using memory model
            pred = functional_call(self.memory_model, params, inputs)
            # Calculate loss, defaults to MSE
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            # Multiply by loss weights
            loss = loss * loss_weights
            return loss.sum()

        # Compute gradients for each sample
        self.per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims = (None, 0, 0, 0))

        # queries for retrieving from the model
        self.to_queries = LinearNoBias(dim, dim_inner) # linear layer mapping feature dim to inner feature dim

        # keys and values for storing to the model
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)  # linear layer mapping feature dim to keys and values
        self.store_memory_loss_fn = store_memory_loss_fn

        # empty memory embed
        # Initialize empty memory embed to zero tensor
        self.empty_memory_embed = nn.Parameter(torch.zeros(dim))
        # Initialize empty memory embed using normal distribution
        nn.init.normal_(self.empty_memory_embed, std = 0.02)

        # learned adaptive learning rate and momentum
        # todo - explore mlp layerwise learned lr / momentum
        self.to_momentum = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # average within chunk
            LinearNoBias(dim, heads),  # linear layer mapping feature dim to number of heads
            Rearrange('b n h -> (b h) n 1')  # reshape tensor
        )

        self.to_adaptive_step = nn.Sequential(
            LinearNoBias(dim, heads),  # linear layer mapping feature dim to number of heads
            Rearrange('b n h -> (b h) n')  # reshape tensor
        )

        self.adaptive_step_transform = adaptive_step_transform

        # allow for softclamp the gradient norms for storing memories
        self.max_grad_norm = max_grad_norm

        # weight decay factor
        self.to_decay_factor = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # average within chunk
            LinearNoBias(dim, heads),  # linear layer mapping feature dim to number of heads
            Rearrange('b n h -> (b h) n 1')  # reshape tensor
        )

        # maybe use accelerated scan
        self.use_accelerated_scan = use_accelerated_scan

    def init_weights_and_momentum(self):
        """
        Initializes weights and momentum for the memory model.

        Returns:
            Tuple[TensorDict, TensorDict]: Initialized weights and momentum as TensorDict objects.
        """
        # Get all parameters of memory model and convert to TensorDict
        params = TensorDict(dict(self.memory_model.named_parameters()))

        # Initialize weights to zero tensors
        init_weights = params.clone().zero_()
        # Initialize momentum to zero tensors
        init_momentum = params.clone().zero_()

        # Return initialized weights and momentum
        return init_weights, init_momentum

    def init_empty_memory_embed(self, batch, seq_len):
        """
        Initializes empty memory embeddings.

        Args:
            batch (int): Batch size.
            seq_len (int): Sequence length.

        Returns:
            Tensor: Initialized empty memory embeddings, shape (batch, seq_len, dim).
        """
        # Repeat empty memory embed to generate tensor of shape (batch, seq_len, dim)
        return repeat(self.empty_memory_embed, 'd -> b n d', b = batch, n = seq_len)

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]]
    ):
        """
        Stores memories and updates weights and momentum of the memory model.

        Args:
            seq (Tensor): Input sequence, shape (batch, seq_len, dim).
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Past state containing weights and momentum.

        Returns:
            Tuple[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]: Updated weights and momentum, and new state.
        """
        # Apply normalization before storage
        seq = self.store_norm(seq)

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk
        seq_len, chunk_size = seq.shape[-2], self.chunk_size
        # Round down sequence length to multiple of chunk size to ensure full chunks
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)

        # Truncate sequence to multiple of chunk size
        seq = seq[:, :round_down_seq_len]

        # curr weights + past weights, in the case that the initial weights are learned
        # Get current weights of memory model
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        # Convert past state to TensorDict
        past_state = tuple(TensorDict(d) for d in past_state)
        past_weights, past_momentum = past_state

        # Add current weights to past weights
        curr_weights = curr_weights + past_weights

        # pack batch and sequence dimension
        # Compute adaptive learning rate:
        # Apply to_adaptive_step to input sequence, then apply adaptive_step_transform
        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        # Compute adaptive momentum:
        # Apply to_momentum to input sequence, then compress to (0, 1) using sigmoid.
        adaptive_momentum = self.to_momentum(seq).sigmoid()

        # Compute weight decay factor:
        # Apply to_decay_factor to input sequence, then compress to (0, 1) using sigmoid.
        decay_factor = self.to_decay_factor(seq).sigmoid()

        # keys and values
        # Separate keys and values:
        # Apply to_keys_values to input sequence, then split into keys and values in last dimension.
        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # maybe multi head
        # Handle multi-head:
        # Apply split_heads to keys and values, merging batch and head dimensions.
        keys, values = map(self.split_heads, (keys, values))

        # Get batch size
        batch = keys.shape[0]

        # take care of chunking
        # Handle chunks:
        # Reshape keys and values in sequence dimension to (batch * n, c, d).
        keys, values = tuple(rearrange(t, 'b (n c) d -> (b n) c d', c = self.chunk_size) for t in (keys, values))

        # Reshape adaptive learning rate:
        # Reshape adaptive learning rate to (batch * n, c) to align with keys and values.
        adaptive_lr = rearrange(adaptive_lr, 'b (n c) -> (b n) c', c = self.chunk_size)

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)
        # Compute gradients and auxiliary loss:
        # Use per_sample_grad_fn to compute gradients for each sample.
        grads = self.per_sample_grad_fn(dict(curr_weights), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # maybe softclamp grad norm
        # Apply soft clamping if max_grad_norm exists
        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension
        # Restore batch and sequence dimensions:
        # Reshape gradient tensors from (batch * n, ...) to (batch, n, ...).
        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # negative gradients, adaptive lr already applied as loss weight
        # Compute surprises:
        # Negative gradients since we are doing gradient descent.
        surprises = grads.apply(lambda t: -t)

        # determine scan function
        # Define default associative scan function:
        # Use associative_scan and binary_operator on input gates and inputs.
        def default_associative_scan(gates, inputs):
            _, outputs = associative_scan(binary_operator, (gates, inputs))
            return outputs

         # If using accelerated scan:
        if self.use_accelerated_scan:
            from accelerated_scan.triton import scan as triton_scan
            from accelerated_scan.warp import scan as warp_scan

            scan = triton_scan if seq.is_cuda else warp_scan

            # Define accelerated scan function:
            # 1. Expand and reshape gates and inputs.
            # 2. Pad sequence length to power of 2.
            # 3. Call scan function.
            # 4. Truncate padded result and restore original shape.
            def accelerate_scan_fn(gates, inputs):
                gates = gates.expand_as(inputs)
                gates, inputs = tuple(rearrange(t, 'b n d -> b d n') for t in (gates, inputs))

                seq_len = gates.shape[-1]
                next_power_two_seq_len = 2 ** max(5, int(math.ceil(math.log2(seq_len))))

                gates = F.pad(gates, (0, next_power_two_seq_len - seq_len))
                inputs = F.pad(inputs, (0, next_power_two_seq_len - seq_len))

                outputs = scan(gates.contiguous(), inputs.contiguous())

                outputs = outputs[..., :seq_len]
                outputs = rearrange(outputs, 'b d n -> b n d')
                return outputs

            scan_fn = accelerate_scan_fn
        else:
            scan_fn = default_associative_scan

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates
        # Compute momentum and updates:
        # 1. Iterate over each parameter name and corresponding surprise.
        # 2. Pack surprise using pack_one_with_inverse and get inverse function.
        # 3. Compute momentum using scan_fn.
        # 4. Compute update using scan_fn (considering weight decay).
        # 5. Unpack update and momentum and store in updates and next_momentum.
        next_momentum = TensorDict()
        updates = TensorDict()

        for param_name, surprise in surprises.items():

            surprise, inverse_pack = pack_one_with_inverse(surprise, 'b n *')

            # derive momentum with associative scan - eq (10)
            # Compute momentum:
            # Use associative scan based on adaptive momentum and surprise.
            momentum = scan_fn(adaptive_momentum, surprise) # momentum is S / surprise in the paper

            # use associative scan again for learned forgetting (weight decay) - eq (13)
            # Compute update:
            # Use associative scan based on decay factor and momentum.
            update = scan_fn(1. - decay_factor, momentum) # momentum is S / surprise in the paper

            updates[param_name] = inverse_pack(update)
            next_momentum[param_name] = inverse_pack(momentum)

        # compute the next weight per batch
        # Compute next weight per batch:
        # Get the last update for each parameter and add to current weight.
        last_update = updates.apply(lambda t: t[:, -1])

        next_state = (curr_weights + last_update, next_momentum)

        return updates, next_state

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
    ):
        """
        Retrieves information from memory.

        Args:
            seq (Tensor): Input sequence, shape (batch, seq_len, dim).
            past_weights (Dict[str, Tensor], optional): Past weights, defaults to None.

        Returns:
            Tensor: Retrieved memory, shape (batch, seq_len + chunk_size - 1, dim).
        """
        # Get chunk size
        chunk_size = self.chunk_size
        # Get batch size and sequence length
        batch, seq_len = seq.shape[:2]

        # Apply normalization before retrieval
        seq = self.retrieve_norm(seq)

        assert seq_len >= chunk_size

        # Truncate sequence, starting from (chunk_size - 1) time step
        seq = seq[:, (chunk_size - 1):]
        # Get truncated sequence length
        curtailed_seq_len = seq.shape[-2]

        # Calculate next sequence length, rounding up to multiple of chunk size
        next_seq_len = round_up_multiple(curtailed_seq_len, chunk_size)

        # Calculate padding length
        padding = next_seq_len - curtailed_seq_len

        # Check if padding is needed
        needs_pad = padding > 0

        if needs_pad:
            # Pad sequence dimension with zeros if needed
            seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper
        # Get current weights of memory model
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        if exists(past_weights):
            # Convert past weights to TensorDict and assert keys match current weights
            past_weights = TensorDict(past_weights)
            assert past_weights.keys() == curr_weights.keys()
            
            # Add current weights to past weights
            curr_weights = curr_weights + past_weights

        # sequence Float['b n d'] to queries
        # Convert sequence from Float['b n d'] to queries
        queries = self.to_queries(seq)

        # maybe multihead
        # Handle multi-head
        queries = self.split_heads(queries)

        # fetch values from memory model
        # Reshape weight tensors to (batch * n, ...) to align with queries
        curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        # Reshape query tensors to (batch * n, c, d) where c is chunk size
        queries = rearrange(queries, 'b (n c) d -> (b n) c d', c = chunk_size)

        # forward functional call
        # Forward pass using memory model to get values
        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # reconstitute batch dimension
        # Restore batch and head dimensions, shape (batch, heads, n * c, d)
        values = rearrange(values, '(b h n) c d -> b h (n c) d', b = batch, h = self.heads)

        # Apply multi-head RMS normalization
        values = self.multihead_rmsnorm(values)

        # maybe gate
        # Apply gating if retrieval gate exists
        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine
        # Merge heads
        values = self.merge_heads(values)

        # Combine heads
        values = self.combine_heads(values)

        # restore, pad with empty memory embed
        # Restore padding:
        # Initialize empty memory embeddings, shape (batch, chunk_size - 1, dim)
        empty_memory_embeds = self.init_empty_memory_embed(values.shape[0], chunk_size - 1)
        # Concatenate empty memory embeddings with retrieved memory, shape (batch, chunk_size, dim)
        values = torch.cat((empty_memory_embeds, values), dim = -2)

        if needs_pad:
            # Remove trailing padding if added earlier
            values = values[:, :-padding]

        # Return retrieved memories
        return values

    def forward(
        self,
        seq,
        store_seq = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        return_next_memories = False
    ):
        """
        Forward pass method.

        Implements memory storage, retrieval, and update processes. Based on input sequence and past state, 
        the model can store new memories, retrieve existing ones, and return current or next memory state.

        Args:
            seq (Tensor): Input sequence, shape (batch, seq_len, dim).
                - `batch`: Batch size.
                - `seq_len`: Sequence length.
                - `dim`: Feature dimension.
            store_seq (Tensor, optional): Sequence for storage, defaults to None.
                - If None, use input sequence `seq` for memory storage.
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]], optional): Past state containing weights and momentum, defaults to None.
                - First dict contains past weights.
                - Second dict contains past momentum.
            return_next_memories (bool, optional): Whether to return next memory state, defaults to False.
                - If True, return updated weights and momentum.
                - If False, return only retrieved memory.

        Returns:
            Tuple[Tensor, Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]]:
                - If `return_next_memories` is False, return retrieved memory, shape (batch, seq_len + chunk_size - 1, dim).
                - If `return_next_memories` is True, return a tuple of retrieved memory and next memory state.
        """
        # Get batch size and sequence length of input sequence
        batch, seq_len = seq.shape[:2]

        if seq_len < self.chunk_size:
            # Return initialized empty memory embeddings if sequence length < chunk size
            return self.init_empty_memory_embed(batch, seq_len)

        if exists(past_state):
            # Convert past state to TensorDict
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            # Initialize weights and momentum if past state doesn't exist
            past_state = self.init_weights_and_momentum()

        # Use input sequence if store_seq not provided
        store_seq = default(store_seq, seq)

        # Store memories and get updates and next memory state
        updates, next_memories = self.store_memories(store_seq, past_state)

        # Get past weights
        past_weights, _ = past_state

        # Retrieve memory: using past weights and updates
        retrieved = self.retrieve_memories(seq, past_weights + updates)

        if not return_next_memories:
            # Return only retrieved memory if next memory state not needed
            return retrieved

        # Return both retrieved memory and next memory state
        return retrieved, next_memories
