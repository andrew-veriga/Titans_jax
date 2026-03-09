from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F


def pad_at_dim(t, pad, dim = -1, value = 0.):
    """
    Pads a tensor at a specified dimension.

    Args:
        t (torch.Tensor): Input tensor to pad.
        pad (Tuple[int, int]): Padding widths, a tuple of two integers representing the number of values to add at the start and end of the specified dimension.
        dim (int, optional): The dimension to pad. Defaults to the last dimension (-1).
        value (float, optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    # Calculate dimension index counting from the right
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)

    # Create a tuple of zeros for dimensions that are not being padded
    zeros = ((0, 0) * dims_from_right)

    # Call F.pad to perform padding
    return F.pad(t, (*zeros, *pad), value = value)


@torch.jit.script
def binary_operator(
    a: tuple[Tensor, Tensor],
    b: tuple[Tensor, Tensor]
):
    """
    Binary operator used in the associative_scan function.

    This operator acts on two input tuples:
    1. Performs element-wise multiplication on the first tensors.
    2. Performs element-wise cumulative multiplication-addition (addcmul) on the second tensors.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): First input tuple containing two tensors.
        b (Tuple[torch.Tensor, torch.Tensor]): Second input tuple containing two tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Result tuple containing two tensors.
    """
    # Unpack the first input tuple
    a_i, kv_i = a
    # Unpack the second input tuple
    a_j, kv_j = b
    # Return the resulting tuple after operations
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)


def associative_scan(
    operator: Callable,
    elems: tuple[Tensor, Tensor]
):
    """
    Performs an associative scan on the input tuples.

    This function implements functionality similar to JAX's lax.associative_scan, 
    specifically designed for handling token sequences in sequence modeling.

    Args:
        operator (Callable): Binary operator function that accepts two input tuples and returns one output tuple.
        elems (Tuple[torch.Tensor, torch.Tensor]): Input tuples containing two tensors of shape (batch_size, sequence_length, ...).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The resulting tuple after the scan.
    """
    # Get sequence length
    num_elems = int(elems[0].shape[1])

    # Check that all input tensors have the same first dimension (sequence length)
    if not all(int(elem.shape[1]) == num_elems for elem in elems[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems]))

    def _scan(elems):
        """
        Performs the scan operation on input tuples.

        Args:
            elems (Tuple[torch.Tensor, torch.Tensor]): Input tuples containing two tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Resulting tuple after the scan.
        """
        # Get sequence length
        num_elems = elems[0].shape[1]

        if num_elems < 2:
            # Return input directly if sequence length is less than 2
            return elems

        # Reduce adjacent pairs of elements
        reduced_elems = operator(
          [elem[:, :-1:2] for elem in elems], # select even-indexed elements
          [elem[:, 1::2] for elem in elems]) # select odd-indexed elements

        # Recursively perform scan on partially reduced tensors
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            # If sequence length is even, merge odd-indexed scan results with original even-indexed elements
            even_elems = operator(
                [e[:, :-1] for e in odd_elems], # select even elements of odd-indexed scan results
                [e[:, 2::2] for e in elems]) # select original even-indexed elements
        else:
            # If sequence length is odd, merge odd-indexed scan results with original even-indexed elements
            even_elems = operator( 
                odd_elems,  # use odd-indexed scan results
                [e[:, 2::2] for e in elems])  # select original even-indexed elements

        # Replace first element of scan result with original first element
        even_elems = [
          torch.cat([elem[:, :1], result], dim=1)
          for (elem, result) in zip(elems, even_elems)]

        # Interleave even and odd indexed scan results
        return list(map(_interleave, even_elems, odd_elems))

    # Perform scan operation and return results
    return _scan(elems)


def _interleave(a, b):
    """
    Interleaves two tensors.

    Args:
        a (torch.Tensor): First input tensor.
        b (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: The interleaved tensor.
    """
    # Get lengths of two tensors along specified dimension
    a_axis_len, b_axis_len = a.shape[1], b.shape[1]
    # Calculate length of output tensor
    output_axis_len = a_axis_len + b_axis_len

    if (a_axis_len == (b_axis_len + 1)):
        # Pad second tensor if first tensor is one element longer
        b = pad_at_dim(b, (0, 1), dim = 1)

    # Stack two tensors along new dimension
    stacked = torch.stack([a, b], dim=2)
    # Flatten stacked tensor to interleave elements
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)

    # Return interleaved tensor truncated to required length
    return interleaved[:, :output_axis_len]
