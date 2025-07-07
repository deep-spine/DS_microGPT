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


import jax.numpy as jnp
import jax

def layer_norm(params, x, eps=1e-5):
    gamma, beta = params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    norm = (x - mean) / (std + eps)
    return gamma * norm + beta

