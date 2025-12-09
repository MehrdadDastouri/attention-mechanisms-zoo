"""Multi-Head Attention implementation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class MultiHeadAttention(BaseAttention):
    """Multi-Head Attention mechanism.
    
    Implements the multi-head attention from "Attention Is All You Need"
    (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
    
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    The computation is:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        
        where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
    
    Each head operates on a subspace of dimension d_k = d_model / num_heads.
    
    Why multiple heads help:
        1. Different heads can learn different types of relationships
        2. Heads can focus on different positions
        3. Provides a form of ensemble within the model
        4. Increases the expressiveness without increasing parameters
    
    Time Complexity: O(n^2 * d)
        - Same as single attention, parallelized across heads
    
    Space Complexity: O(n^2 * h + d^2)
        - n^2 * h for attention weights per head
        - d^2 for projection matrices
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head (d_model // num_heads).
        w_q: Query projection matrix.
        w_k: Key projection matrix.
        w_v: Value projection matrix.
        w_o: Output projection matrix.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        """Initialize Multi-Head Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads. Must divide d_model evenly.
            dropout: Dropout probability applied to attention weights.
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
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
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
        """Compute multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional attention mask of shape (batch_size, 1, 1, seq_len_k)
                or (batch_size, 1, seq_len_q, seq_len_k). Positions with True/1
                are masked (not attended to).
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Attention distribution of shape 
                (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections: (batch, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct shape for broadcasting
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked rows
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute attended values: (batch, num_heads, seq_len_q, d_k)
        attended = torch.matmul(attention_weights, v)
        
        # Reshape back: (batch, seq_len_q, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len_q, self.d_model)
        
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
            f"d_k={self.d_k}, dropout={self.dropout.p}"
        )
