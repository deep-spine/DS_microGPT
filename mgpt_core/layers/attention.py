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


import jax 
import jax.numpy as jnp
import haiku as hk
from mgpt_core.layers.rotary_positional_embedding import get_freqs, RoPE

__all__ = ["MhAttention"]

def split_heads(x:jnp.ndarray, n_heads:int):
    batch, seq, dim = x.shape
    n_dim = dim // n_heads
    x = x.reshape(batch, seq, n_heads, n_dim)
    x = jnp.transpose(x, (0, 2, 1, 3))
    return x

def merge_heads(x):
    batch, n_heads, seq, n_dim = x.shape
    x = jnp.transpose(x, (0, 2, 1, 3))
    x = x.reshape(batch, seq, n_heads * n_dim)
    return x


class MhAttention(hk.Module):

    """
    Multi-head self-attention module implemented in Haiku.

    This module performs scaled dot-product attention over multiple heads, with optional causal masking.
    It's a core component of Transformer-based architectures.

    Args:
        d_model (int): Total dimensionality of the input and output vectors (model width).
        n_heads (int): Number of attention heads.
        mask (bool): If True, applies causal masking to prevent attending to future tokens (default: False).
        name (str): Optional name for the module (default: "attention").

    Attributes:
        d_model (int): Dimensionality of model input/output.
        n_heads (int): Number of attention heads.
        mask (bool): Whether causal masking is enabled.

    Notes:
        - Internally splits `d_model` across `n_heads` to compute queries, keys, and values.
        - Attention output is projected back to `d_model` after concatenation.
        - Causal masking is typically used during autoregressive generation.
    """

    def __init__(self, d_model, n_heads, mask=False, name="attention"):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.mask = mask

    def __call__(self, x: jnp.ndarray):
        batch, seq, dim = x.shape
        n_dim = dim // self.n_heads

        def safe_initializer():
            base_init = hk.initializers.VarianceScaling(
                scale=0.01,
                mode='fan_avg',
                distribution='uniform'
            )
            return lambda shape, dtype: base_init(shape, jnp.float32)

        init = safe_initializer()

        qw = hk.get_parameter('qw', shape=[self.d_model, self.d_model], init=init)
        kw = hk.get_parameter('kw', shape=[self.d_model, self.d_model], init=init)
        vw = hk.get_parameter('vw', shape=[self.d_model, self.d_model], init=init)
        ow = hk.get_parameter('ow', shape=[self.d_model, self.d_model], init=init)

        # Project to Q, K, V in float32 for numerical stability
        Q = (x @ qw).astype(jnp.float32)
        K = (x @ kw).astype(jnp.float32)
        V = (x @ vw).astype(jnp.float32)

        Q = Q.reshape(batch, seq, self.n_heads, n_dim)
        K = K.reshape(batch, seq, self.n_heads, n_dim)
        freq = get_freqs(seq, n_dim)
        Q = RoPE(Q, freq)
        K = RoPE(K, freq)

        Q = jnp.transpose(Q, (0, 2, 1, 3))  
        K = jnp.transpose(K, (0, 2, 1, 3)) 
        V = split_heads(V, self.n_heads)   

        # Attention score: [B, H, T, T]
        scale = jnp.sqrt(n_dim).astype(jnp.float32)
        score = (Q @ jnp.swapaxes(K, -1, -2)) / scale

        if self.mask:
            mask = jnp.tril(jnp.ones((1, 1, seq, seq), dtype=jnp.float32))
            score = score - 1e10 * (1.0 - mask)

        score = score - jnp.max(score, axis=-1, keepdims=True)
        weights = jax.nn.softmax(score, axis=-1)
        attn = weights @ V  

        merged = merge_heads(attn) 
        out = (merged @ ow).astype(x.dtype)  # Cast back to input dtype

        return out