import math

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class LinearBERTSelfAttention(BertSelfAttention):
    def __init__(self, attn_type="inter-word",  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_type = attn_type
        if self.attn_type == "mixture":
            ones = torch.zeros(1, self.num_attention_heads, 1)
            self.weight = nn.Parameter(
                ones,
                requires_grad=False
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.attn_type == "inter-word":
            # [batch_size, n_heads, q_seq_len, k_seq_len]

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
        elif self.attn_type == "inter-hidden":
            # [batch_size, n_heads, q_seq_len, k_seq_len]

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(key_layer.transpose(-1, -2), value_layer)

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-2)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(query_layer, attention_probs)
        elif self.attn_type == "mixture":
            # [batch_size, n_heads, q_seq_len, k_seq_len]

            attention_scores_iw = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores_ih = torch.matmul(key_layer.transpose(-1, -2), value_layer)

            attention_scores_iw = attention_scores_iw / math.sqrt(self.attention_head_size)
            attention_scores_ih = attention_scores_ih / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                attention_scores_iw = attention_scores_iw + attention_mask

            attention_probs_iw = nn.functional.softmax(attention_scores_iw, dim=-1)
            attention_probs_ih = nn.functional.softmax(attention_scores_ih, dim=-1)

            attention_probs_iw = self.dropout(attention_probs_iw)
            attention_probs_ih = self.dropout(attention_probs_ih)

            if head_mask is not None:
                attention_probs_iw = attention_probs_iw * head_mask
                attention_probs_ih = attention_probs_ih * head_mask

            context_layer_iw = torch.matmul(attention_probs_iw, value_layer)
            context_layer_ih = torch.matmul(query_layer, attention_probs_ih)

            context_layer = (1 - torch.sigmoid(self.weight)) * context_layer_iw + \
                            torch.sigmoid(self.weight) * context_layer_ih
        else:
            raise ValueError("Unknown attn type")

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
