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


from ds_mgpt.seed import KEYS
import haiku as hk
import jax.random as random
from nami.JAX import Nami

class FFN_Nami(hk.Module):
    def __init__(self, out_feat, name = "ffn_nami"):
        super().__init__(name = name)
        self.out_feat = out_feat

    def __call__(self, x):
        fc = hk.Linear(output_size = self.out_feat * 4)
        gate = hk.Linear(output_size = self.out_feat * 4) 
        proj = hk.Linear(output_size = self.out_feat)
        nami = Nami()

        _x = fc(x)
        x_gate = nami(gate(x))
        x_proj = proj(_x * x_gate)

        return x_proj
    
