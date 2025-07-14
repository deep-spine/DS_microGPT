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


from mgpt_core.layers.attention import MhAttention
from mgpt_core.layers.ffn import FFN_Nami
import haiku as hk
import jax
import jax.numpy as jnp

class Transformer(hk.Module):
    def __init__(self, dim:int, heads:int, mask=False,name = "transformer"):
        super().__init__(name=name)
        self.LN1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.LN2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.Attn = MhAttention(dim, heads, mask=mask)
        self.FFN = FFN_Nami(dim)
    
    def __call__(self, x):
        x = x + self.Attn(self.LN1(x))
        x = x + self.FFN(self.LN2(x))

        return x


class Hyuga_neuro05(hk.Module):
    def __init__(
        self,
        vocab_size: int,
        mask=True,
        name="Hyuga_neuro05"
    ):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.dim = 1024
        self.heads = 32
        self.num_layers = 42
        self.mask = mask

        self.embedding_weights = hk.get_parameter(
            "embed", shape=[vocab_size, self.dim], init=hk.initializers.TruncatedNormal(stddev=0.02)
        )

        self.blocks = [
            Transformer(self.dim, self.heads, self.mask, name=f"transformer_{i}") for i in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.take(self.embedding_weights, x, axis=0) 

        for block in self.blocks:
            x = block(x)

        # output projection: tie weights by using embedding matrix transposed
        logits = x @ self.embedding_weights.T
        return logits


class Hyuga_echo(hk.Module):
    def __init__(
        self,
        vocab_size: int,
        mask=True,
        name="Hyuga_echo"
    ):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.dim = 2048
        self.heads = 32
        self.num_layers = 24
        self.mask = mask

        self.embedding_weights = hk.get_parameter(
            "embed", shape=[vocab_size, self.dim], init=hk.initializers.TruncatedNormal(stddev=0.02)
        )

        self.blocks = [
            Transformer(self.dim, self.heads, self.mask, name=f"transformer_{i}") for i in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.take(self.embedding_weights, x, axis=0) 

        for block in self.blocks:
            x = block(x)

        # output projection: tie weights by using embedding matrix transposed
        logits = x @ self.embedding_weights.T
        return logits


# from mgpt_core.seed import KEYS
# k1, k2, k3 = KEYS(42, 3)
# def f(x):
#     t = Hyuga_neuro05(
#         vocab_size=50304,
#         mask=True
#         )
#     return t(x)

# net = hk.transform(f)
# x = jax.random.randint(k1, (1, 3072), 1, 1000)

# param = net.init(k2, x)

# # out = net.apply(param, k3, x)

# print(x.shape)



# def count_params(params):
#     return sum(jnp.size(x) for x in jax.tree_util.tree_leaves(params))

# print(count_params(param)*1e-6) 

# del param, x, net
