from mgpt_legacy.decoder.embed.embedding import *

def test()->None:
    out = pos_encoding(10, 6)
    assert out.shape == (10, 6)