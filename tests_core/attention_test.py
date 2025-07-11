from mgpt_core.layers.attention import MhAttention
import jax.random as random
import jax.numpy as jnp
import haiku as hk

def test():
    from mgpt_core.seed import KEYS
    def fn(x):
        mha = MhAttention(512, 8)
        return mha(x)
    
    k1, k2 = KEYS(split=2)
    x = random.normal(k1, (1, 256, 512))
    f = hk.transform(fn)
    param = f.init(k2, x)
    y = f.apply(param, k2, x)
    
    assert x.shape == y.shape
