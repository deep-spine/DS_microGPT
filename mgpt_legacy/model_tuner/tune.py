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


class microLM:
    def __init__(self, model, params):
        self.model = model
        self.loaded_params = params
        self.layer_names = list(self.loaded_params.keys())
    
    def parameters(self, freeze:list=[]):
        if freeze == []:
            return self.loaded_params
        else:
            params = {}
            for p in self.layer_names:
                if not p in freeze:
                    params[f'{p}'] = self.loaded_params[p]
            return params

def LMrun(model_fn, X, params:dict, params_new:dict = {}, num_heads=8):
    p = params.copy()
    if params_new != {}:
        for key in params_new.keys():
            p[key] = params_new[key]
    return model_fn(X, p, num_heads)
