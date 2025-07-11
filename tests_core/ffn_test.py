from mgpt_core.layers.ffn import FFN_Nami
import jax.random as random
import haiku as hk

def test()->None:
    from mgpt_core.seed import KEYS
    k1, k2, k3 = KEYS(42, 3)
    def f(x):
        ffn = FFN_Nami(out_feat = 64*4)
        return ffn(x)
    
    net = hk.transform(f)
    x = random.normal(k1, (1, 32, 64))
    param = net.init(k2, x)
    
    out = net.apply(param, k3, x)
    assert out.shape == (1, 32, 256)