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


import jax.random as random

def KEYS(seed = 0, split=None):
    key = random.PRNGKey(seed)
    args = key
    if not split == None:
        args = random.split(key, split)
    return args

