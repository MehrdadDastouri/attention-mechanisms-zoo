"""Cross-Attention implementation for encoder-decoder architectures."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class CrossAttention(BaseAttention):
    """Cross-Attention mechanism for encoder-decoder models.
    
    Cross-attention allows the decoder to attend to all positions in the
    encoder output. This is essential for sequence-to-sequence tasks like
    machine translation, image captioning, and speech recognition.
    
    In cross-attention:
        - Query (Q) comes from the decoder
        - Key (K) and Value (V) come from the encoder
    
    The computation follows:
        CrossAttention(Q_dec, K_enc, V_enc) = softmax(Q_dec @ K_enc^T / sqrt(d_k)) @ V_enc
    
    Use Cases:
        1. Machine Translation: Decoder attends to source sentence
        2. Image Captioning: Language model attends to image features
        3. Speech Recognition: Text decoder attends to audio features
        4. Visual Question Answering: Question attends to image
    
    Time Complexity: O(n * m * d)
        - n: decoder sequence length (queries)
        - m: encoder sequence length (keys/values)
        - d: model dimension
    
    Space Complexity: O(n * m)
        - Attention weight matrix size
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head.
        w_q: Query projection (from decoder).
        w_k: Key projection (from encoder).
        w_v: Value projection (from encoder).
        w_o: Output projection.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        """Initialize Cross-Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
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
        
        # Query projection (from decoder)
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        
        # Key and Value projections (from encoder)
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
        """Compute cross-attention.
        
        Args:
            query: Decoder query tensor of shape (batch_size, seq_len_q, d_model).
            key: Encoder key tensor of shape (batch_size, seq_len_k, d_model).
            value: Encoder value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional padding mask for encoder of shape 
                (batch_size, 1, 1, seq_len_k). True positions are masked.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Cross-attention distribution of shape 
                (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Project queries from decoder
        q = self.w_q(query)
        
        # Project keys and values from encoder
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply encoder padding mask if provided
        if mask is not None:
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
        
        # Reshape: (batch, seq_len_q, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.w_o(attended)
        
        return output, attention_weights
    
    @property
    def complexity(self) -> str:
        """Return time complexity string.
        
        Returns:
            Time complexity as "O(n * m * d)" where n is query length,
            m is key length, and d is model dimension.
        """
        return "O(n * m * d)"
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"d_k={self.d_k}, dropout={self.dropout.p}"
        )
