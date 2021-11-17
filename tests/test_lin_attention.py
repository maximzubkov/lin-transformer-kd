from typing import Union

import torch

from modules.linear_attention import LinearAttention, CausalLinearAttention


@torch.no_grad()
def _run_attention(attn: Union[LinearAttention, CausalLinearAttention]):
    attn.eval()

    seq_len = 16
    batch_size = 32
    embed_dim = 24
    num_heads = 8

    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    k = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    v_1, _ = attn.forward(q, k, v, output_attention=True)
    v_2, _ = attn.forward(q, k, v, output_attention=False)
    assert torch.allclose(v_1, v_2, atol=1e-4)


@torch.no_grad()
def test_lin_attn():
    attn = LinearAttention()
    _run_attention(attn)


@torch.no_grad()
def test_causal_lin_attn():
    attn = CausalLinearAttention()
    _run_attention(attn)
