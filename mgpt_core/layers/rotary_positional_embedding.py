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

def get_freqs(seq_len, dim, base=10000):
    
    """
    Generates rotary positional frequencies for use in RoPE.

    Args:
        seq_len (int): Sequence length.
        dim (int): Embedding dimension (must be even).
        base (float): Base value for frequency scaling (default: 10000).

    Returns:
        jnp.ndarray: A [seq_len, dim // 2] array containing positional frequencies.
    """

    half_dim = dim // 2
    freq = 1.0 / (base ** (jnp.arange(0, half_dim) / half_dim))
    pos = jnp.arange(seq_len)
    freqs = jnp.outer(pos, freq) 
    return freqs

def RoPE(x, freqs):

    """
    Applies Rotary Positional Embedding (RoPE) to input tensor using given frequencies.

    Args:
        x (jnp.ndarray): Input tensor of shape [batch, seq_len, heads, dim].
        freqs (jnp.ndarray): Positional frequency matrix from `get_freqs`, shape [seq_len, dim // 2].

    Returns:
        jnp.ndarray: Tensor with the same shape as `x`, with rotary position applied to final dimension.
    """

    cos = jnp.cos(freqs)[None, :, None, :]  
    sin = jnp.sin(freqs)[None, :, None, :] 

    x1, x2 = jnp.split(x, 2, axis=-1) 
    x_rotated = jnp.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    return x_rotated



import torch

def get_freqs_t(seq_len, dim, base=10000, device=None):

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
