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


import torch
import torch.nn as nn
from nami.Torch import Nami 
from mgpt_core.layers.rotary_positional_embedding import get_freqs_t, RoPE_t


def split_heads(x, n_heads):
    batch, seq, dim = x.shape
    n_dim = dim // n_heads
    x = x.view(batch, seq, n_heads, n_dim).transpose(1, 2)
    return x

def merge_heads(x):
    batch, n_heads, seq, n_dim = x.shape
    x = x.transpose(1, 2).contiguous().view(batch, seq, n_heads * n_dim)
    return x


class MhAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask=False, device = torch.device('cuda')):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mask = mask
        self.device = device

        self.qw = nn.Parameter(torch.randn(d_model, d_model))
        self.kw = nn.Parameter(torch.randn(d_model, d_model))
        self.vw = nn.Parameter(torch.randn(d_model, d_model))
        self.ow = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        B, S, D = x.shape
        n_dim = D // self.n_heads

        Q = x @ self.qw
        K = x @ self.kw
        V = x @ self.vw

        Q = Q.view(B, S, self.n_heads, n_dim)
        K = K.view(B, S, self.n_heads, n_dim)

        freqs = get_freqs_t(S, n_dim, device=self.device)
        Q = RoPE_t(Q, freqs)
        K = RoPE_t(K, freqs)

        Q = Q.permute(0, 2, 1, 3)  # (B, H, S, D)
        K = K.permute(0, 2, 1, 3)
        V = split_heads(V, self.n_heads)

        scores = Q @ K.transpose(-1, -2) / self.d_model**0.5
        scores = scores.float()

        if self.mask:
            mask = torch.tril(torch.ones(S, S, dtype=torch.float32, device=x.device))
            scores = scores.masked_fill(mask[None, None, :, :] == 0, -1e10)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        attn = weights @ V
        out = merge_heads(attn) @ self.ow
        return out


class FFN_Nami(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 4)
        self.gate = nn.Linear(dim, dim * 4)
        self.proj = nn.Linear(dim * 4, dim)
        self.nami = Nami()

    def forward(self, x):
        _x = self.fc(x)
        x_gate = self.nami(self.gate(x))
        return self.proj(_x * x_gate)


class Transformer(nn.Module):
    def __init__(self, dim, heads, mask=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = MhAttention(dim, heads, mask=mask)
        self.ffn = FFN_Nami(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Hyuga_neuro05(nn.Module):
    def __init__(self, vocab_size, mask=True):
        super().__init__()
        self.dim = 1024
        self.heads = 32
        self.num_layers = 42
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, self.dim)
        self.transformers = nn.ModuleList([
            Transformer(self.dim, self.heads, mask=mask) for _ in range(self.num_layers)
        ])

        self.out_proj = lambda x: x @ self.embed.weight.T

    def forward(self, x):
        x = self.embed(x)
        for block in self.transformers:
            x = block(x)
        return self.out_proj(x)


class Hyuga_echo(nn.Module):
    def __init__(self, vocab_size, mask=True):
        super().__init__()
        self.dim = 2048
        self.heads = 32
        self.num_layers = 24
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, self.dim)
        self.transformers = nn.ModuleList([
            Transformer(self.dim, self.heads, mask=mask) for _ in range(self.num_layers)
        ])

        self.out_proj = lambda x: x @ self.embed.weight.T

    def forward(self, x):
        x = self.embed(x)
        for block in self.transformers:
            x = block(x)
        return self.out_proj(x)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = torch.compile(Hyuga_echo(vocab_size=52000, mask = True))
# print(model)
# print(f"Total trainable parameters: {count_parameters(model):,}")

# del model