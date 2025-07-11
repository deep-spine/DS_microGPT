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
    def __init__(self, d_model, n_heads, mask = False, name="attention"):
        super().__init__(name = name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.mask = mask
    
    def __call__(self, x:jnp.ndarray):

        batch, seq, dim = x.shape
        n_dim = dim // self.n_heads

        qw = hk.get_parameter('qw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        kw = hk.get_parameter('kw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        vw = hk.get_parameter('vw', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())
        ow = hk.get_parameter('ow', shape=[self.d_model, self.d_model], init=hk.initializers.RandomNormal())

        Q = x @ qw
        K = x @ kw
        V = x @ vw

        # split_heads fuction is useless for Q and K since RoPE has to be implemented 
        # between reshape and transpose
        Q = Q.reshape(batch, seq, self.n_heads, n_dim)
        K = K.reshape(batch, seq, self.n_heads, n_dim)
        freq = get_freqs(seq, n_dim)
        Q = RoPE(Q, freq)
        K = RoPE(K, freq)

        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = split_heads(V, self.n_heads)

        score = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(self.d_model)
        score = score.astype(jnp.float32)

        if self.mask:
            _, _, seq_l, _ = x.shape
            mask = jnp.tril(jnp.ones((1, 1, seq_l, seq_l), dtype=jnp.float32))
            score = score - 1e10 * (1.0 - mask)

        weights = jax.nn.softmax(score, axis=-1)
        attn = weights @ V

        merged = merge_heads(attn)
        out = merged @ ow
        return out
    

