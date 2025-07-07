import jax.random as random

def KEYS(seed = 0, split=None):
    key = random.PRNGKey(seed)
    args = key
    if not split == None:
        args = random.split(key, split)
    return args

