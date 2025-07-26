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


from mgpt_core.seed import KEYS
import haiku as hk
import jax.random as random
import jax.numpy as jnp
from nami.JAX import Nami

class FFN_Nami(hk.Module):

    """
    Feedforward network (FFN) block with Nami activation gating.

    This module implements a Transformer-style FFN with a gated activation mechanism,
    where the gating path uses the custom Nami activation function. The architecture
    resembles Gated Linear Units (GLUs) or SwiGLU but uses Nami for potentially smoother 
    gradient flow and improved expressivity.

    Architecture:
        x -> Linear (4 * out_feat) ---> (_x)
        x -> Linear (4 * out_feat) -> Nami() ---> (gate)
        output = proj(_x * gate)

    Args:
        out_feat (int): Output feature dimension of the FFN block.
        name (str): Optional name for the module (default: "ffn_nami").

    Attributes:
        out_feat (int): Dimensionality of the final output after projection.

    Notes:
        - The intermediate hidden size is expanded to 4× `out_feat`.
        - Gated activations improve expressiveness over standard MLPs.
        - This FFN block is typically used after attention in a Transformer layer.
    """

    def __init__(self, out_feat, name = "ffn_nami"):
        super().__init__(name = name)
        self.out_feat = out_feat

    def __call__(self, x):
        initializer = hk.initializers.VarianceScaling(
            scale=1.0,
            mode='fan_avg',
            distribution='uniform',
            dtype=jnp.float16
        )

        fc = hk.Linear(output_size=self.out_feat * 4, w_init=initializer, b_init=jnp.zeros)
        gate = hk.Linear(output_size=self.out_feat * 4, w_init=initializer, b_init=jnp.zeros)
        proj = hk.Linear(output_size=self.out_feat, w_init=initializer, b_init=jnp.zeros)

        nami = Nami()  # assuming it's your custom activation (like GELU, SwiGLU etc.)

        _x = fc(x)
        x_gate = nami(gate(x))
        x_proj = proj(_x * x_gate)

        return x_proj

