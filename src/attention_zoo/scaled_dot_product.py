"""Scaled Dot-Product Attention implementation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class ScaledDotProductAttention(BaseAttention):
    """Scaled Dot-Product Attention mechanism.
    
    Implements the core attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
    
    The attention function is computed as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    where:
        - Q: Query matrix of shape (batch, seq_len_q, d_k)
        - K: Key matrix of shape (batch, seq_len_k, d_k)
        - V: Value matrix of shape (batch, seq_len_k, d_v)
        - d_k: Dimension of keys (used for scaling)
    
    The scaling factor 1/sqrt(d_k) is applied to prevent the dot products
    from growing too large in magnitude, which would push the softmax
    function into regions with extremely small gradients.
    
    Time Complexity: O(n^2 * d)
        - Computing QK^T: O(n^2 * d)
        - Softmax: O(n^2)
        - Matrix multiplication with V: O(n^2 * d)
    
    Space Complexity: O(n^2)
        - Storing the attention weight matrix
    
    Attributes:
        d_model: Model dimension (embedding size).
        scale: Scaling factor (1/sqrt(d_model)).
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        """Initialize Scaled Dot-Product Attention.
        
        Args:
            d_model: The dimensionality of the model.
            dropout: Dropout probability applied to attention weights.
        """
        super().__init__(d_model, dropout)
        self.scale = 1.0 / math.sqrt(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional attention mask of shape (batch_size, 1, seq_len_q, seq_len_k)
                or (batch_size, seq_len_q, seq_len_k). Positions with True/1 are masked.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Attention distribution of shape 
                (batch_size, seq_len_q, seq_len_k).
        """
        # Compute attention scores: (batch, seq_len_q, seq_len_k)
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension for compatibility
            scores = scores.masked_fill(mask.bool().squeeze(1), float("-inf"))
        
        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked rows
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values: (batch, seq_len_q, d_model)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    @property
    def complexity(self) -> str:
        """Return time complexity string.
        
        Returns:
            Time complexity as "O(n^2 * d)" where n is sequence length
            and d is model dimension.
        """
        return "O(n^2 * d)"
