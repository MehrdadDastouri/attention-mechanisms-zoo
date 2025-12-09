"""Grouped-Query Attention implementation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class GroupedQueryAttention(BaseAttention):
    """Grouped-Query Attention (GQA) mechanism.
    
    From "GQA: Training Generalized Multi-Query Transformer Models from 
    Multi-Head Checkpoints" (Ainslie et al., 2023):
    https://arxiv.org/abs/2305.13245
    
    GQA interpolates between MHA and MQA by grouping query heads to share
    K/V heads. With G groups:
    - G = H (num_heads): equivalent to MHA
    - G = 1: equivalent to MQA
    
    Used in LLaMA 2/3, Mistral, and other modern LLMs.
    
    Time Complexity: O(n^2 * d)
    Space Complexity: O(n * d_k * G) for KV cache
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
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
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K/V for each group
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        return self.w_o(attended), attn_weights
    
    @property
    def complexity(self) -> str:
        return "O(n^2 * d)"
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}"
