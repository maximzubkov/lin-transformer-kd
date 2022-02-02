from transformers import GPT2LMHeadModel, BertForSequenceClassification

from models.linear_gpt2 import LinearGPT2Attention
from models.linear_bert import LinearBERTSelfAttention


def make_attention_linear(model, feature_map=None):
    if isinstance(model, GPT2LMHeadModel):
        for i, _ in enumerate(model.transformer.h):
            layer = model.transformer.h[i]
            layer.attn = LinearGPT2Attention(feature_map, model.config, is_cross_attention=False)
            if hasattr(layer, "crossattention"):
                layer.crossattention = LinearGPT2Attention(feature_map, model.config, is_cross_attention=True)
    return model


def update_attn(model, is_inter_word: bool = True):
    if isinstance(model, BertForSequenceClassification):
        attn_type = "inter-word" if is_inter_word else "inter-hidden"
        print(f"Update, {attn_type}")
        for i, _ in enumerate(model.bert.encoder.layer):
            layer = model.bert.encoder.layer[i]
            layer.attention.self = LinearBERTSelfAttention(attn_type, model.config)
            if hasattr(layer, "crossattention"):
                layer.attention.self = LinearBERTSelfAttention(is_inter_word, model.config, position_embedding_type="absolute")
    return model
