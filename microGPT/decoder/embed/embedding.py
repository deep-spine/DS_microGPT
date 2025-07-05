import jax
import jax.numpy as jnp
from jax import lax

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
    out = out.reshape(seq_len, d_model)
    return out


def _embed(params, token_idx):
    emb = params["embedding_table"][token_idx]
    d_model = params["embedding_table"].shape[1]
    return emb * jnp.sqrt(d_model)


def word_embedding(params, tokens):
    emb_table = params["embedding_table"]
    d_model = emb_table.shape[1]

    embeddings = emb_table[tokens] * jnp.sqrt(d_model)

    seq_len = tokens.shape[1]
    MAX_SEQ_LEN = 2048

    full_pos = pos_encoding(MAX_SEQ_LEN, d_model)

    pos = lax.slice(
        full_pos,
        start_indices=(0, 0),
        limit_indices=(seq_len, d_model)
    )

    pos = jnp.expand_dims(pos, axis=0)  # (1, seq_len, d_model)

    return embeddings + pos
