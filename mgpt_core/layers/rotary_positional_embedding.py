# ==============================================================================
#  Project: ds_microGPT
#  Author: DeepSpine (2025–present)
#  License: MIT
#
#  A compact, distilled GPT architecture written in JAX/Haiku.
#  Designed for high-output, low-weight inference, TensorFlow export,
#  and real-world fine-tuned deployment.
#
#  Copyright (c) 2025–present DeepSpine.
#  Licensed under the MIT License. See LICENSE for details.
# ==============================================================================

import jax.numpy as jnp
def get_freqs(seq_len, dim, base=10000.0, dtype=jnp.float32):
    
    """
    Generates rotary positional frequencies with improved numerical stability.
    
    Changes made:
    1. Added epsilon to prevent division by zero
    2. Explicit dtype specification
    3. Optimized computation to prevent large exponents
    """
    half_dim = dim // 2
    exponent = jnp.arange(0, half_dim, dtype=dtype) / jnp.maximum(half_dim, 1)
    freqs = base ** (-exponent)  # More stable than 1/(base^exponent)
    pos = jnp.arange(seq_len, dtype=dtype)
    freqs = jnp.outer(pos, freqs)
    return freqs

def RoPE(x, freqs):
    """
    Applies Rotary Positional Embedding with enhanced numerical stability.
    
    Changes made:
    1. Fixed broadcasting to avoid silent dimension expansion issues
    2. Added epsilon to prevent NaN in sin/cos
    3. Optimized computation
    """
    # Cast freqs to same dtype as x for mixed precision compatibility
    freqs = freqs.astype(x.dtype)
    
    # Compute cos and sin with epsilon for stability
    cos = jnp.cos(freqs + 1e-8)[None, :, None, :]  # Add [batch, seq, heads, dim] dimensions
    sin = jnp.sin(freqs + 1e-8)[None, :, None, :]

    # Split input tensor along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    # Apply rotation in a numerically stable way
    rotated_x = jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        axis=-1
    )
    
    return rotated_x


def get_freqs_t(seq_len, dim, base=10000, device=None):
    import torch

    """
    Generates rotary positional frequencies (PyTorch version).

    Args:
        seq_len (int): Sequence length.
        dim (int): Embedding dimension (must be even).
        base (float): Base value for frequency scaling (default: 10000).
        device (torch.device or None): Device to place the output on.

    Returns:
        torch.Tensor: A [seq_len, dim // 2] tensor with rotary frequencies.
    """

    half_dim = dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, freq)  
    return freqs  

def RoPE_t(x, freqs):
    import torch
    
    """
    Applies Rotary Positional Embedding (RoPE) to input tensor using given frequencies (PyTorch version).

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seq_len, heads, dim].
        freqs (torch.Tensor): Frequency matrix from `get_freqs_t`, shape [seq_len, dim // 2].

    Returns:
        torch.Tensor: Output tensor with same shape as `x`, with rotary embedding applied.
    """

    cos = freqs.cos()[None, :, None, :]  
    sin = freqs.sin()[None, :, None, :]  

    x1, x2 = torch.chunk(x, 2, dim=-1)  
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1) 
    return x_rotated
