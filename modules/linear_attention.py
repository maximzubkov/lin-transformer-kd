import torch
from torch.nn import Module

from .feature_maps import fm

class LinearAttention(Module):
    def __init__(self, feature_map: str = None, eps: float = 1e-10):
        super(LinearAttention, self).__init__()
        self.feature_map = fm(feature_map)
        self.eps = eps

    def _linear(self, Q, K, V):
        # [batch_size, n_heads, p_s, p_s]
        KV = torch.einsum("nshd,nshm->nhmd", K, V)
        
        # Z equals to denominator value after applying attention
        # [batch_size, target_seq_len, n_heads]
        Z = 1 / torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous(), None

    def _quadratic(self, Q, K, V):
        # [batch_size, n_heads, input_seq_len, target_seq_len]
        QK = torch.einsum("nlhe,nshe->nhls", Q, K)

        A = QK / (torch.sum(QK, dim=-1, keepdim=True) + self.eps)
        V = torch.einsum("nhls,nshd->nlhd", A, V)

        return V.contiguous(), A

    def forward(self, queries, keys, values, output_attention: bool = False):
        # [batch_size, q_seq_len, n_heads, p_s]
        Q = self.feature_map(queries)

        # [batch_size, k_seq_len, n_heads, p_s]
        K = self.feature_map(keys)
            
        if output_attention:    
            V, A = self._quadratic(Q, K, values)
        else:
            V, A = self._linear(Q, K, values) 

        return V, A 


class CausalLinearAttention(Module):
    def __init__(self, feature_map: str = None, eps: float = 1e-10):
        super(CausalLinearAttention, self).__init__()
        self.feature_map = fm(feature_map)
        self.eps = eps

    def _linear(self, Q, K, V):
        # [batch_size, seq_len, n_heads, p_s, p_s]
        KV = torch.einsum("nshd,nshm->nshmd", K, V)

        # [batch_size, n_heads, p_s]
        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        
        # [batch_size, seq_len, n_heads, p_s]
        V = torch.einsum("nlhd,nlhmd,nlh->nlhm", Q, KV.cumsum(1), Z)

        return V.contiguous(), None

    def _quadratic(self, Q, K, V):
        # [batch_size, n_heads, input_seq_len, target_seq_len]
        QK = torch.einsum("nlhe,nshe->nhls", Q, K)
        QK = QK * torch.tril(torch.ones(QK.shape[2:]))

        A = QK / (torch.sum(QK, dim=-1, keepdim=True) + self.eps)
        V = torch.einsum("nhls,nshd->nlhd", A, V)

        return V.contiguous(), A

    def forward(self, queries, keys, values, output_attention: bool = False):
        # [batch_size, q_seq_len, n_heads, p_s]
        Q = self.feature_map(queries)

        # [batch_size, k_seq_len, n_heads, p_s]
        K = self.feature_map(keys)
            
        if output_attention:    
            V, A = self._quadratic(Q, K, values)
        else:
            V, A = self._linear(Q, K, values) 

        return V, A
