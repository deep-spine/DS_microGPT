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
    
    """
    Single Transformer encoder block using Haiku, composed of LayerNorm, Multi-head Attention with RoPE, 
    and a gated FFN block with Nami activation.

    This module applies:
        - Pre-layer normalization before attention and feedforward.
        - Multi-head self-attention using rotary positional embeddings (RoPE).
        - Feedforward network (FFN) with gating based on the custom Nami activation.
        - Residual connections after both attention and FFN components.

    Args:
        dim (int): Hidden size of the model (embedding dimension).
        heads (int): Number of attention heads.
        mask (bool): Whether to apply causal masking for autoregressive models (default: False).
        name (str): Optional name for this module instance (default: "transformer").

    Attributes:
        LN1 (hk.LayerNorm): Layer normalization before attention.
        LN2 (hk.LayerNorm): Layer normalization before feedforward.
        Attn (MhAttention): Multi-head self-attention with RoPE and optional masking.
        FFN (FFN_Nami): Feedforward network using Nami-activated gating.

    Input:
        x (jnp.ndarray): Input tensor of shape [batch, seq_len, dim].

    Returns:
        jnp.ndarray: Output tensor of the same shape as input.
    """

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


class Hyuga_neuro(hk.Module):

    """
    Hyuga_neuro is a compact yet deep language model optimized for reasoning and high-throughput inference.

    Model Architecture:
    - Parameters: ~756M
    - Hidden Size (d_model): 1024
    - FFN Inner Dimension: 4096
    - Attention Heads: 32
    - Layers: 42
    - Positional Encoding: Rotary (RoPE)
    - Activation: Nami (custom smooth nonlinearity)

    Attributes:
        vocab_size (int): Size of the input vocabulary.
        dim (int): Embedding dimension and model width.
        heads (int): Number of self-attention heads.
        num_layers (int): Total number of Transformer layers.
        mask (bool): Whether to use causal attention masking.

    Notes:
        - Uses weight tying between input embedding and output projection layer.
        - Designed for research in efficient transformer reasoning.

    """

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
        self.final_ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)


        self.blocks = [
            Transformer(self.dim, self.heads, self.mask, name=f"transformer_{i}") for i in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.take(self.embedding_weights, x, axis=0) 

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        # output projection: tie weights by using embedding matrix transposed
        logits = x @ self.embedding_weights.T
        return logits


class Hyuga_echo(hk.Module):

    """
    Hyuga_echo is a powerful language model designed for high-capacity reasoning and dialogue tasks, 
    balancing depth and width for efficient inference at scale.

    Model Architecture:
    - Parameters: ~1.7B
    - Hidden Size (d_model): 2048
    - FFN Inner Dimension: 8192
    - Attention Heads: 32
    - Layers: 24
    - Positional Encoding: Rotary (RoPE)
    - Activation Function: Nami (smooth and efficient nonlinearity)

    Attributes:
        vocab_size (int): Size of the model vocabulary.
        dim (int): Embedding dimension and transformer model width.
        heads (int): Number of attention heads used in multi-head self-attention.
        num_layers (int): Number of stacked Transformer blocks.
        mask (bool): Enables causal masking for autoregressive generation.

    Notes:
        - Uses weight tying between input and output embeddings.
        - Part of the Hyuga series, optimized for strong generalization and decoding performance.
    """

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
        self.final_ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.blocks = [
            Transformer(self.dim, self.heads, self.mask, name=f"transformer_{i}") for i in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.take(self.embedding_weights, x, axis=0) 

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        # output projection: tie weights by using embedding matrix transposed
        logits = x @ self.embedding_weights.T
        return logits


# Usage:
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
