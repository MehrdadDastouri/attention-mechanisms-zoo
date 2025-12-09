"""Causal (Masked) Attention implementation for autoregressive models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class CausalAttention(BaseAttention):
    """Causal (Masked) Attention mechanism for autoregressive generation.
    
    Causal attention ensures that each position can only attend to previous
    positions (including itself), which is essential for autoregressive
    language modeling (GPT-style models).
    
    The causal mask is a lower triangular matrix:
        
        Position:  0  1  2  3
        Query 0:  [1  0  0  0]
        Query 1:  [1  1  0  0]
        Query 2:  [1  1  1  0]
        Query 3:  [1  1  1  1]
    
    Where 1 means "can attend" and 0 means "masked".
    
    Implementation Details:
        - The causal mask is precomputed and registered as a buffer
        - Efficient implementation using upper triangular masking
        - Compatible with KV caching for fast inference
    
    Time Complexity: O(n^2 * d)
        - Same as standard attention, but ~half the actual computation
          due to the triangular structure
    
    Space Complexity: O(n^2)
        - Full attention matrix is still computed (just masked)
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head.
        max_seq_len: Maximum sequence length for the causal mask buffer.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        bias: bool = True
    ) -> None:
        """Initialize Causal Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for precomputed mask.
            bias: Whether to include bias in linear projections.
        
        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.max_seq_len = max_seq_len
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Precompute causal mask
        # True values indicate positions that should be masked (not attended to)
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        if self.w_q.bias is not None:
            nn.init.zeros_(self.w_q.bias)
            nn.init.zeros_(self.w_k.bias)
            nn.init.zeros_(self.w_v.bias)
            nn.init.zeros_(self.w_o.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute causal attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
                For self-attention, key = query.
            value: Value tensor of shape (batch_size, seq_len, d_model).
                For self-attention, value = query.
            mask: Optional additional mask (e.g., padding mask) of shape
                (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len).
                Will be combined with causal mask using logical OR.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len, d_model).
            attention_weights: Causal attention distribution of shape 
                (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float("-inf"))
        
        # Apply additional mask if provided (e.g., padding mask)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked rows
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute attended values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape: (batch, seq_len, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.w_o(attended)
        
        return output, attention_weights
    
    @property
    def complexity(self) -> str:
        """Return time complexity string.
        
        Returns:
            Time complexity as "O(n^2 * d)" where n is sequence length
            and d is model dimension.
        """
        return "O(n^2 * d)"
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"d_k={self.d_k}, max_seq_len={self.max_seq_len}, "
            f"dropout={self.dropout.p}"
        )
