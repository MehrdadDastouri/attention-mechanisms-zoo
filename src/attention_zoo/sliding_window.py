"""Sliding Window Attention implementation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class SlidingWindowAttention(BaseAttention):
    """Sliding Window Attention mechanism.
    
    From "Longformer: The Long-Document Transformer" (Beltagy et al., 2020):
    https://arxiv.org/abs/2004.05150
    
    Each position attends only to positions within a fixed-size window,
    enabling linear complexity in sequence length.
    
    Time Complexity: O(n * w * d) where w is window size
    Space Complexity: O(n * w)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        return torch.abs(diff) > self.window_size
    
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
        k = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sliding window mask
        window_mask = self._create_window_mask(seq_len_q, query.device)
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
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
        return "O(n * w * d)"
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, window_size={self.window_size}"
