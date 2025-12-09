"""Multi-Query Attention implementation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class MultiQueryAttention(BaseAttention):
    """Multi-Query Attention (MQA) mechanism.
    
    From "Fast Transformer Decoding: One Write-Head is All You Need" 
    (Shazeer, 2019): https://arxiv.org/abs/1911.02150
    
    Multiple query heads with single shared K/V head. Reduces memory
    bandwidth for inference, especially with KV caching.
    
    Time Complexity: O(n^2 * d)
    Space Complexity: O(n^2 + n*d_k) - reduced KV cache
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads")
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, self.d_k, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key)
        v = self.w_v(value)
        
        scores = torch.matmul(q, k.unsqueeze(1).transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, v.unsqueeze(1))
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        return self.w_o(attended), attn_weights
    
    @property
    def complexity(self) -> str:
        return "O(n^2 * d)"
