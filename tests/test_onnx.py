from typing import Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

from modules.linear_attention import LinearAttention, CausalLinearAttention


@torch.no_grad()
def _convert_to_onnx_quadratic(attn: Union[LinearAttention, CausalLinearAttention], model_name: str):
    attn.eval()

    seq_len = 16
    batch_size = 32
    embed_dim = 24
    num_heads = 8

    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    k = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    v_orig, attn_orig = attn.forward(q, k, v, output_attention=True)

    # Export the model
    torch.onnx.export(
        attn,
        (q, k, v, True),
        model_name,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['q', 'k', 'v', 'output_attention'],
        output_names=['values', 'attention'],
    )
    model = onnx.load(model_name)
    onnx.checker.check_model(model)

    ort_session = ort.InferenceSession(model_name)

    v_onnx, attn_onnx = ort_session.run(
        None,
        {
            "q": q.cpu().numpy(),
            "k": k.cpu().numpy(),
            "v": v.cpu().numpy(),
        },
    )
    assert np.all(np.isclose(v_onnx, v_orig.cpu().numpy(), atol=1e-3))
    assert np.all(np.isclose(attn_onnx, attn_orig.cpu().numpy(), atol=1e-3))


@torch.no_grad()
def _convert_to_onnx_linear(attn: Union[LinearAttention, CausalLinearAttention], model_name: str):
    attn.eval()

    seq_len = 16
    batch_size = 32
    embed_dim = 24
    num_heads = 8

    q = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    k = torch.rand(batch_size, seq_len, num_heads, embed_dim)

    v_orig, _ = attn.forward(q, k, v, output_attention=False)

    # Export the model
    torch.onnx.export(
        attn,
        (q, k, v, False),
        model_name,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['q', 'k', 'v', 'output_attention'],
        output_names=['values'],
    )
    model = onnx.load(model_name)
    onnx.checker.check_model(model)

    ort_session = ort.InferenceSession(model_name)

    v_onnx = ort_session.run(
        None,
        {
            "q": q.cpu().numpy(),
            "k": k.cpu().numpy(),
            "v": v.cpu().numpy(),
        },
    )
    assert np.all(np.isclose(v_onnx, v_orig.cpu().numpy(), atol=1e-3))


@torch.no_grad()
def test_lin_attn():
    attn = LinearAttention()
    _convert_to_onnx_quadratic(attn, model_name="linear-attn-naive.onnx")
    _convert_to_onnx_linear(attn, model_name="linear-attn.onnx")


@torch.no_grad()
def test_causal_lin_attn():
    attn = CausalLinearAttention()
    _convert_to_onnx_quadratic(attn, model_name="linear-attn-naive.onnx")
    _convert_to_onnx_linear(attn, model_name="causal-linear-attn.onnx")
