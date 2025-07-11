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
    half_dim = dim // 2
    freq = 1.0 / (base ** (jnp.arange(0, half_dim) / half_dim))
    pos = jnp.arange(seq_len)
    freqs = jnp.outer(pos, freq) 
    return freqs

def RoPE(x, freqs):
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
    half_dim = dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, freq)  
    return freqs  

def RoPE_t(x, freqs):
    cos = freqs.cos()[None, :, None, :]  
    sin = freqs.sin()[None, :, None, :]  

    x1, x2 = torch.chunk(x, 2, dim=-1)  
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1) 
    return x_rotated
