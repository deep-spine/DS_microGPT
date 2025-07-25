import jax.random as random
from mgpt_legacy.decoder.attn.attention import *

def test()->None:
    key = random.PRNGKey(0)
    x, k1 = random.split(key)
    X = random.normal(x, (1, 6, 4))
    params = [random.normal(k1, (4, 4)) for _ in range(4)]
    y = multi_head_attention(params, X, 4)
    assert y.shape == (1, 6, 4)

