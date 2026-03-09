import math
from functools import partial
import einx
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange, Reduce

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad
from tensordict import TensorDict

from associative_scan import associative_scan, binary_operator, pad_at_dim


# Use partial to create a linear layer function without bias
LinearNoBias = partial(Linear, bias = False)


"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
"""


def exists(v):
    """
    Check if a variable exists (is not None).

    Args:
        v: Any variable.

    Returns:
        bool: True if v is not None, False otherwise.
    """
    return v is not None


def default(v, d):
    """
    Returns the variable itself if it exists (is not None); otherwise, returns a default value.

    Args:
        v: Any variable.
        d: Default value.

    Returns:
        Any: v if it exists; otherwise, d.
    """
    return v if exists(v) else d


def round_down_multiple(seq, mult):
    """
    Rounds down an integer to the nearest multiple of a specified integer.

    Args:
        seq (int): The integer to round.
        mult (int): The multiple.

    Returns:
        int: The rounded-down integer.
    """
    return seq // mult * mult


def round_up_multiple(seq, mult):
    """
    Rounds up an integer to the nearest multiple of a specified integer.

    Args:
        seq (int): The integer to round.
        mult (int): The multiple.

    Returns:
        int: The rounded-up integer.
    """
    return math.ceil(seq / mult) * mult


def pack_one_with_inverse(t, pattern):
    """
    Packs a tensor according to a specified pattern and returns an inverse function for unpacking.

    Args:
        t (torch.Tensor): The tensor to pack.
        pattern (Tuple[int, ...]): Packing pattern, specifying how each dimension is split.

    Returns:
        Tuple[torch.Tensor, Callable]: The packed tensor and an inverse function for unpacking.
    """
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


class MemoryAttention(Module):
    """
    Temporal attention as a memory module.

    This module uses self-attention as a memory access method, calculating hidden representations through query, key, and value weights.
    """
    def __init__(
        self,
        dim
    ):
        """
        Initializes the MemoryAttention module.

        Args:
            dim (int): Feature dimension.
        """
        super().__init__()
        # Define four learnable weight parameters for queries, keys, values weight 1, and values weight 2
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # queries
            nn.Parameter(torch.randn(dim, dim)), # keys
            nn.Parameter(torch.randn(dim, dim)), # values weight 1
            nn.Parameter(torch.randn(dim, dim)), # values weight 2
        ])

    def forward(self, x):
        """
        Forward pass method.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        assert x.shape[-2] > 1, 'chunk size needs to be greater than 1 for using attention as memory'

        # Unpack weight parameters
        wq, wk, wv1, wv2 = self.weights

        # Calculate queries, keys, and values, shape (b, n, d)
        q = x @ wq
        k = x @ wk
        v = x @ wv1

        # Use scaled dot-product attention to compute hidden representation
        # is_causal=True ensures current time step only sees past time steps
        hidden = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        # Apply SiLU activation to hidden representation and multiply by values weight 2 for final output
        return F.silu(hidden) @ wv2  # output shape (b, n, d)


def default_loss_fn(pred, target):
    """
    Default loss function: Mean Squared Error (MSE) loss.

    Args:
        pred (Tensor): Predicted values, shape (batch_size, ...).
        target (Tensor): Target values, same shape as pred.

    Returns:
        Tensor: Computed loss, shape (batch_size,).
    """
    return (pred - target).pow(2).mean(dim = -1).sum()


class NeuralMemory(Module):
    """
    Neural Memory Module.

    This module enhances model capability through a memory mechanism, allowing the model to store and retrieve information when processing sequence data.
    The memory mechanism is based on the MemoryAttention model and uses gradient descent to update memory content.
    """
    def __init__(
        self,
        dim,
        chunk_size = 1,
        dim_head = None,
        heads = 1,
        model: MemoryAttention | None = None,
        store_memory_loss_fn = default_loss_fn,
        pre_rmsnorm = True,
        post_rmsnorm = True,
        use_accelerated_scan = False,
        default_model_kwargs: dict = dict()
    ):
        """
        Initializes the Neural Memory Module.

        Args:
            dim (int): Feature dimension.
            chunk_size (int, optional): Chunk size for grouping, defaults to 1.
            dim_head (int, optional): Dimension of each attention head, defaults to dim if None.
            heads (int, optional): Number of attention heads, defaults to 1.
            model (MemoryAttention, optional): Memory model, defaults to None. If None, default MemoryAttention model is used.
            store_memory_loss_fn (Callable, optional): Loss function for storing memory, defaults to MSE.
            pre_rmsnorm (bool, optional): Whether to apply RMSNorm before storage and retrieval, defaults to True.
            post_rmsnorm (bool, optional): Whether to apply RMSNorm after storage and retrieval, defaults to True.
            use_accelerated_scan (bool, optional): Whether to use accelerated scan, defaults to False.
            default_model_kwargs (dict, optional): Default keyword arguments for memory model, defaults to empty dict.
        """
        super().__init__()

        # Define normalization layers
        # Normalization before retrieval
        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # Normalization before storage
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        # Normalization after retrieval or storage
        self.post_rmsnorm = nn.RMSNorm(dim) if post_rmsnorm else nn.Identity()

        # Handle multi-head attention
        # Default dim_head to dim if not specified
        dim_head = default(dim_head, dim)
        # Calculate inner dimension
        dim_inner = dim_head * heads

        # Reshape tensor to multi-head form
        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)
        # Merge multi-head tensor back to original shape
        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)
        # If multiple heads, use linear layer to merge; otherwise, use identity mapping
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        # Initialize memory model
        if not exists(model):
            # Use default MemoryAttention model if none provided
            model = MemoryAttention(dim_head, **default_model_kwargs)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # Assign model to memory_model
        self.memory_model = model

        # Save chunk size
        self.chunk_size = chunk_size

        # Define forward pass and loss calculation function
        def forward_and_loss(params, inputs, target):
            # Call memory model with params for forward pass
            pred = functional_call(self.memory_model, params, inputs)
            # Calculate loss, defaults to MSE
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            return loss

        # Compute gradients for each sample
        self.per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims = (None, 0, 0))

        # Define query linear layer for retrieval
        self.to_queries = LinearNoBias(dim, dim_inner)

        # Define key and value linear layers for storage
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)
        # Save loss function
        self.store_memory_loss_fn = store_memory_loss_fn

        # Define modules for computing adaptive learning rate and momentum
        self.to_momentum = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # average over intra-chunk dimension
            LinearNoBias(dim, heads),  # linear layer mapping dimension to number of attention heads
            Rearrange('b n h -> (b h) n 1')  # reshape tensor
        )

        self.to_adaptive_step = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # average over intra-chunk dimension
            LinearNoBias(dim, heads),  # linear layer mapping dimension to number of attention heads
            Rearrange('b n h -> (b h) n')  # reshape tensor
        )

        # Define module for computing weight decay factor
        self.to_decay_factor = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),  # average over intra-chunk dimension
            LinearNoBias(dim, heads),  # linear layer mapping dimension to number of attention heads
            Rearrange('b n h -> (b h) n 1')  # reshape tensor
        )

        # Whether to use accelerated scan
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

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]]
    ):
        """
        Stores memories and updates weights and momentum of the memory model.

        Args:
            seq (torch.Tensor): Input sequence, shape (batch_size, seq_len, dim).
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]]): Past state containing weights and momentum.

        Returns:
            Tuple[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]]]: Updated weights and momentum, and new state.
        """
        # Apply normalization before storage
        seq = self.store_norm(seq)

        # Calculate sequence length and chunk size
        seq_len, chunk_size = seq.shape[-2], self.chunk_size
        # Round down sequence length to multiple of chunk size to ensure full chunks
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)

        # Truncate sequence to multiple of chunk size
        seq = seq[:, :round_down_seq_len]

        # Get current weights of memory model
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        # Convert past state to TensorDict objects
        past_state = tuple(TensorDict(d) for d in past_state)
        past_weights, past_momentum = past_state

        # Add current weights to past weights
        curr_weights = curr_weights + past_weights

        # Pack batch and sequence dimensions
        # 'b' for batch, 'n' for sequence length, 'c' for intra-chunk dimension

        # Compute adaptive learning rate:
        # 1. Apply to_adaptive_step to input sequence, result shape (batch, n, heads).
        # 2. Compress to (0, 1) using sigmoid.
        # 3. Multiply by -15 and exponentiate to map to (1e-7, 1).
        adaptive_lr = (self.to_adaptive_step(seq).sigmoid() * -15).exp() # learning rate ranges from 1 to 1e-7

        # Compute adaptive momentum:
        # Apply to_momentum to input sequence, then compress to (0, 1) using sigmoid.
        adaptive_momentum = self.to_momentum(seq).sigmoid()

        # Compute weight decay factor:
        # Apply to_decay_factor to input sequence, then compress to (0, 1) using sigmoid.
        decay_factor = self.to_decay_factor(seq).sigmoid()

        # Separate keys and values:
        # Apply to_keys_values to input sequence, then split into keys and values in last dimension.
        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # Handle multi-head:
        # Apply split_heads to keys and values, merging batch and head dimensions.
        keys, values = map(self.split_heads, (keys, values))

        # Get batch size
        batch = keys.shape[0]

        # Handle chunks:
        # Reshape keys and values in sequence dimension to (batch * n, c, d).
        keys, values = tuple(rearrange(t, 'b (n c) d -> (b n) c d', c = self.chunk_size) for t in (keys, values))

        # Compute gradients and auxiliary loss:
        # Use per_sample_grad_fn to compute gradients for each sample using current weights, keys, and values.
        grads = self.per_sample_grad_fn(dict(curr_weights), keys, values)

        # Convert gradients to TensorDict
        grads = TensorDict(grads)

        # Restore batch and sequence dimensions:
        # Reshape gradient tensors from (batch * n, ...) to (batch, n, ...).
        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # Multiply by adaptive learning rate:
        # Apply multiplication operation to each gradient tensor, multiplying gradient by negative learning rate.
        surprises = grads.apply(lambda t: einx.multiply('b n ..., b n -> b n ...', t, -adaptive_lr))

        # Define default associative scan function:
        # Use associative_scan and binary_operator on input gates and inputs.
        def default_associative_scan(gates, inputs):
            _, outputs = associative_scan(binary_operator, (gates, inputs))
            return outputs

        # If using accelerated scan
        if self.use_accelerated_scan:
            # Import scan functions from triton and accelerated_scan modules
            from triton import scan as triton_scan
            from accelerated_scan import scan as warp_scan

            # Select scan function based on device
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

                outputs = scan(gates, inputs)

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
        # For each parameter, get the last update and add to current weight.
        last_update = updates.apply(lambda t: t[:, -1])

        next_state = (curr_weights + last_update, next_momentum)
        
        # Return updated weights and momentum, and new state
        return updates, next_state

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
    ):
        """
        Retrieves information from memory.

        Args:
            seq (torch.Tensor): Input sequence, shape (batch_size, seq_len, dim).
            past_weights (Dict[str, Tensor], optional): Past weights used for memory retrieval.

        Returns:
            torch.Tensor: Retrieved memory, shape (batch_size, seq_len - chunk_size, dim).
        """
        # Get chunk size
        chunk_size = self.chunk_size
        # Get sequence length
        seq_len = seq.shape[1]

        # Apply normalization before retrieval
        seq = self.retrieve_norm(seq)

        assert seq_len > chunk_size

        # Truncate sequence, starting from chunk_size time step
        seq = seq[:, chunk_size:]
        # Get truncated sequence length
        curtailed_seq_len = seq.shape[-2]

        # Calculate sequence length for next chunk, rounding up to multiple of chunk size
        next_seq_len = round_up_multiple(curtailed_seq_len + 1, chunk_size)

        # Calculate padding length
        padding = next_seq_len - curtailed_seq_len

        # Pad sequence dimension with zeros
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # Memory model parameters store key/value memories
        # When MLP has 1 weight matrix, it is equivalent to `kv` fast weight memory in linear attention (retrieval is q @ (kv))

        # Get current weights of memory model
        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        if exists(past_weights):
            # Convert past weights to TensorDict
            past_weights = TensorDict(past_weights)
            assert past_weights.keys() == curr_weights.keys()

            # Add current weights to past weights
            curr_weights = curr_weights + past_weights

        # Convert sequence from Float['b n d'] to queries
        # Apply query linear layer to sequence to generate queries
        queries = self.to_queries(seq)

        # Handle multi-head
        # Apply multi-head reshape to queries
        queries = self.split_heads(queries)

        # Get batch size
        batch = queries.shape[0]

        # Fetch values from memory model
        # Reshape weight tensors to (batch * n, ...)
        curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        # Reshape query tensors to (batch * n, c, d)
        queries = rearrange(queries, 'b (n c) d -> (b n) c d', c = chunk_size)

        # Forward functional call
        # Call memory model with current weights and queries to get values
        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # Restore batch dimension
        # Reshape value tensors to (batch, n * c, d)
        values = rearrange(values, '(b n) c d -> b (n c) d', b = batch)

        # Merge heads and combine
        # Apply multi-head merge to values
        values = self.merge_heads(values)
        # Apply multi-head combination to values
        values = self.combine_heads(values)

        # Post normalization
        # Added for training stability, though not mentioned in paper
        values = self.post_rmsnorm(values)

        # Restore padding
        # Pad sequence dimension with zeros (todo: use learned empty memory embedding instead of 0)
        values = pad_at_dim(values, (chunk_size, 0), dim = 1, value = 0.) 
        # Truncate padded sequence, removing trailing padding
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

        Args:
            seq (torch.Tensor): Input sequence, shape (batch_size, seq_len, dim).
            store_seq (torch.Tensor, optional): Sequence used for storage, defaults to None. If None, input sequence is used.
            past_state (Tuple[Dict[str, Tensor], Dict[str, Tensor]], optional): Past state containing weights and momentum.
            return_next_memories (bool, optional): Whether to return next memory state, defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor]]]]: Retrieved memory and optional next memory state.
        """
        # Get batch size and sequence length
        batch, seq_len = seq.shape[:2]

        if seq_len <= self.chunk_size:
            # If sequence length <= chunk size, return zero tensor
            return torch.zeros_like(seq)

        if exists(past_state):
            # Convert past state to TensorDict
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            # If past state doesn't exist, initialize weights and momentum
            past_state = self.init_weights_and_momentum()

        # If store_seq not provided, use input sequence
        store_seq = default(store_seq, seq)

        # Call store_memories to get updates and next memory state
        updates, next_memories = self.store_memories(store_seq, past_state)

        # Get past weights
        past_weights, _ = past_state

        # Call retrieve_memories to get retrieved memory
        retrieved = self.retrieve_memories(seq, past_weights + updates)

        if not return_next_memories:
            # If not returning next memory state, return retrieved memory
            return retrieved

        # Return retrieved memory and next memory state
        return retrieved, next_memories
