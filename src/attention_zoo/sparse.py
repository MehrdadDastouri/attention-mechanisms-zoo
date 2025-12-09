"""Sparse Attention implementation from Sparse Transformer."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class SparseAttention(BaseAttention):
    """Sparse Attention mechanism from Sparse Transformer.
    
    Implements sparse attention patterns from "Generating Long Sequences 
    with Sparse Transformers" (Child et al., 2019): 
    https://arxiv.org/abs/1904.10509
    
    Instead of full O(n^2) attention, sparse attention uses structured
    sparsity patterns to achieve O(n * sqrt(n)) complexity. This enables
    processing of much longer sequences.
    
    Sparsity Patterns:
        1. Strided Pattern: Each position attends to every k-th position
           (where k = sqrt(n)), capturing long-range dependencies.
        
        2. Fixed Pattern: Each position attends to a local window of
           size k = sqrt(n), capturing local dependencies.
        
        3. Combined: Alternating layers use strided vs fixed patterns,
           or both patterns are combined in the same layer.
    
    Example (k=4, n=16):
        Strided: position i attends to {0, 4, 8, 12} + local block
        Fixed: position i attends to positions in same block of size 4
    
    Time Complexity: O(n * sqrt(n) * d)
        - Each position attends to O(sqrt(n)) other positions
    
    Space Complexity: O(n * sqrt(n))
        - Sparse attention weight storage
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head.
        block_size: Size of local attention blocks.
        pattern: Sparsity pattern ('strided', 'fixed', or 'combined').
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        block_size: int = 64,
        pattern: str = "combined",
        num_global_tokens: int = 0,
        bias: bool = True
    ) -> None:
        """Initialize Sparse Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            block_size: Size of local attention blocks. Typically sqrt(n).
            pattern: Sparsity pattern - 'strided', 'fixed', or 'combined'.
            num_global_tokens: Number of tokens that attend to all positions
                (useful for CLS tokens).
            bias: Whether to include bias in linear projections.
        
        Raises:
            ValueError: If pattern is not one of the supported patterns.
        """
        super().__init__(d_model, dropout)
        
        if pattern not in ("strided", "fixed", "combined"):
            raise ValueError(f"pattern must be 'strided', 'fixed', or 'combined', got {pattern}")
        
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.block_size = block_size
        self.pattern = pattern
        self.num_global_tokens = num_global_tokens
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
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
    
    def _create_sparse_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sparse attention mask based on the selected pattern.
        
        Args:
            seq_len: Sequence length.
            device: Device to create mask on.
        
        Returns:
            Boolean mask of shape (seq_len, seq_len) where True means
            the position should be masked (not attended to).
        """
        # Start with all positions masked
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        
        if self.pattern in ("fixed", "combined"):
            # Local block attention: each position attends to its block
            for i in range(seq_len):
                block_start = (i // self.block_size) * self.block_size
                block_end = min(block_start + self.block_size, seq_len)
                mask[i, block_start:block_end] = False
        
        if self.pattern in ("strided", "combined"):
            # Strided attention: attend to every block_size-th position
            for i in range(seq_len):
                for j in range(0, seq_len, self.block_size):
                    mask[i, j] = False
                # Also attend to last position of each previous block
                for j in range(self.block_size - 1, seq_len, self.block_size):
                    if j < seq_len:
                        mask[i, j] = False
        
        # Global tokens attend to and are attended by all
        if self.num_global_tokens > 0:
            mask[:self.num_global_tokens, :] = False
            mask[:, :self.num_global_tokens] = False
        
        return mask
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
            value: Value tensor of shape (batch_size, seq_len, d_model).
            mask: Optional additional mask to combine with sparse mask.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len, d_model).
            attention_weights: Sparse attention distribution of shape 
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
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create and apply sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, query.device)
        scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Apply additional mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
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
            Time complexity as "O(n * sqrt(n) * d)".
        """
        return "O(n * sqrt(n) * d)"
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"block_size={self.block_size}, pattern={self.pattern}, "
            f"num_global_tokens={self.num_global_tokens}"
        )
