# ==============================================================================
#  Project: ds_microGPT
#  Author: DeepSpine Research (2025–present)
#  License: MIT
#
#  A compact, distilled GPT architecture written in JAX/Haiku.
#  Designed for high-output, low-weight inference, TensorFlow export,
#  and real-world fine-tuned deployment.
#
#  Copyright (c) 2025–present DeepSpine Research.
#  Licensed under the MIT License. See LICENSE for details.
# ==============================================================================


import jax
import jax.numpy as jnp
from jax import lax
from gpt_jaxfn.decoder.params.param_setup import init_embedding_params

def pos_encoding(seq_len, d_model):
    i = jnp.arange(d_model)
    even_i = i[::2]
    denominator = jnp.power(10000.0, even_i / d_model)

    def body_fn(pos, _):
        position = jnp.array(pos, dtype=jnp.float32)
        angle_args = position / denominator
        row = jnp.zeros(d_model)

        row = row.at[::2].set(jnp.sin(angle_args))
        row = row.at[1::2].set(jnp.cos(angle_args))
        return pos + 1, row

    _, out = lax.scan(body_fn, 0, None, length=seq_len)
    out = out = out.reshape(seq_len, d_model)
    return out

pos_encoding = jax.jit(pos_encoding, static_argnames=('d_model', 'seq_len'))


def _embed(params, token_idx):
    emb = params["embedding_table"][token_idx]
    d_model = params["embedding_table"].shape[1]
    return emb * jnp.sqrt(d_model)

def word_embedding(params, tokens):
    emb_table = params["embedding_table"] 
    d_model = emb_table.shape[1]

    embeddings = emb_table[tokens] * jnp.sqrt(d_model)

    seq_len = tokens.shape[1]
    pos = pos_encoding(seq_len, d_model)  # (seq_len, d_model)
    pos = jnp.expand_dims(pos, axis=0)    # (1, seq_len, d_model)

    return embeddings + pos 


