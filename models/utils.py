from transformers import GPT2LMHeadModel

from models.linear_gpt2 import LinearGPT2Attention


def make_attention_linear(model, feature_map=None):
    if isinstance(model, GPT2LMHeadModel):
        for i, _ in enumerate(model.transformer.h):
            layer = model.transformer.h[i]
            layer.attn = LinearGPT2Attention(feature_map, model.config, is_cross_attention=False)
            if hasattr(layer, "crossattention"):
                layer.crossattention = LinearGPT2Attention(feature_map, model.config, is_cross_attention=True)
    return model
