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

    """
    Multi-head self-attention module with Rotary Positional Embedding (RoPE) and optional causal masking.

    This implementation performs multi-head attention using manually defined query, key, value,
    and output projection matrices (`qw`, `kw`, `vw`, `ow`). Rotary positional encoding is applied 
    to both Q and K tensors before attention computation.

    Args:
        d_model (int): Total hidden dimension of the model (embedding size).
        n_heads (int): Number of attention heads.
        mask (bool): If True, applies causal (autoregressive) masking (default: False).
        device (torch.device): Device for positional frequency generation (default: CUDA).

    Attributes:
        qw (nn.Parameter): Query projection weight of shape [d_model, d_model].
        kw (nn.Parameter): Key projection weight of shape [d_model, d_model].
        vw (nn.Parameter): Value projection weight of shape [d_model, d_model].
        ow (nn.Parameter): Output projection weight of shape [d_model, d_model].

    Input:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].

    Notes:
        - Q and K are positionally encoded using Rotary Positional Embedding (RoPE).
        - Attention weights are computed via scaled dot-product attention.
        - If `mask=True`, causal masking is applied using a lower triangular mask.
        - Uses helper functions `get_freqs_t`, `RoPE_t`, `split_heads`, and `merge_heads`.
    """

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

    """
    Feedforward network (FFN) block with Nami-activated gating.

    This module implements a gated feedforward block for Transformers. It expands the input using
    a linear layer, applies the Nami activation to a parallel gating path, multiplies them element-wise, 
    and then projects back to the original dimension.

    Architecture:
        x -> Linear (4 * dim) ---> (_x)
        x -> Linear (4 * dim) -> Nami() ---> (gate)
        output = proj(_x * gate)

    Args:
        dim (int): Input and output feature dimension.

    Attributes:
        fc (nn.Linear): Main linear layer that expands input to 4 * dim.
        gate (nn.Linear): Parallel linear layer whose output is gated via Nami activation.
        proj (nn.Linear): Final projection layer to reduce back to `dim`.
        nami (nn.Module): Custom Nami activation function.

    Returns:
        torch.Tensor: Output tensor of shape [batch, seq_len, dim].
    """

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
    
    """
    Single Transformer encoder block using LayerNorm, Multi-head Attention with RoPE, and FFN_Nami.

    This block applies:
        - Pre-layer normalization before attention and feedforward.
        - Multi-head attention with Rotary Positional Embeddings (RoPE).
        - Feedforward network using Nami-activated gating.
        - Residual connections after both attention and FFN sublayers.

    Args:
        dim (int): Hidden size of the model (embedding dimension).
        heads (int): Number of attention heads.
        mask (bool): If True, enables causal masking (for autoregressive models).

    Attributes:
        ln1 (nn.LayerNorm): Layer normalization before attention.
        ln2 (nn.LayerNorm): Layer normalization before feedforward.
        attn (MhAttention): Multi-head self-attention module with RoPE and optional masking.
        ffn (FFN_Nami): Feedforward network with Nami activation gating.

    Input:
        x (torch.Tensor): Input tensor of shape [batch_size, seq_len, dim].

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_len, dim].
    """

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


class Hyuga(nn.Module):

    """
    Hyuga_neuro is a compact yet deep language model optimized for reasoning and high-throughput inference.

    Model Architecture:
    - Parameters: ~362M
    - Hidden Size (d_model): 512
    - FFN Inner Dimension: 2048
    - Attention Heads: 8
    - Layers: 80
    - Positional Encoding: Rotary (RoPE)
    - Activation: Nami (custom smooth nonlinearity)

    Attributes:
        vocab_size (int): Size of the input vocabulary.
        dim (int): Embedding dimension and model width.
        heads (int): Number of self-attention heads.
        num_layers (int): Total number of Transformer layers.
        mask (bool): Whether to use causal attention masking.

    Notes:
        - Uses weight tying between input embedding and output projection layer.
        - Designed for research in efficient transformer reasoning.

    """

    def __init__(self, vocab_size, mask=True):
        super().__init__()
        self.dim = 512
        self.heads = 8
        self.num_layers = 80
        self.vocab_size = vocab_size
        

        self.embed = nn.Embedding(vocab_size, self.dim)
        self.transformers = nn.ModuleList([
            Transformer(self.dim, self.heads, mask=mask) for _ in range(self.num_layers)
        ])
        self.final_ln = nn.LayerNorm(self.dim)

        self.out_proj = lambda x: x @ self.embed.weight.T

    def forward(self, x):
        x = self.embed(x)
        for block in self.transformers:
            x = block(x)
        x = self.final_ln(x) 
        return self.out_proj(x)



# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = torch.compile(Hyuga_neuro(vocab_size=52000, mask = True))
# print(model)
# print(f"Total trainable parameters: {count_parameters(model):,}")

# del model