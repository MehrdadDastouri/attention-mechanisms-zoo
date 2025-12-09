"""Linear Attention implementation using kernel feature maps."""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map for linear attention.
    
    This ensures non-negative outputs, which is required for the
    kernel interpretation of attention.
    
    Args:
        x: Input tensor.
    
    Returns:
        ELU(x) + 1, ensuring positive values.
    """
    return F.elu(x) + 1


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ReLU feature map for linear attention.
    
    Simple non-negative feature map using ReLU.
    
    Args:
        x: Input tensor.
    
    Returns:
        ReLU(x).
    """
    return F.relu(x)


class LinearAttention(BaseAttention):
    """Linear Attention mechanism using kernel feature maps.
    
    Implements linear attention from "Transformers are RNNs: Fast 
    Autoregressive Transformers with Linear Attention" 
    (Katharopoulos et al., 2020): https://arxiv.org/abs/2006.16236
    
    Standard attention computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V
    
    The softmax can be viewed as a kernel: softmax(qk^T) = exp(q)exp(k)^T
    
    Linear attention uses a feature map phi to approximate this:
        LinearAttention(Q, K, V) = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
    
    Key insight: By changing the order of operations from (QK^T)V to Q(K^T V),
    we reduce complexity from O(n^2 * d) to O(n * d^2).
    
    This is beneficial when the sequence length n is much larger than the
    feature dimension d.
    
    Feature Map Choices:
        1. ELU + 1: phi(x) = ELU(x) + 1 (ensures positivity)
        2. ReLU: phi(x) = ReLU(x)
        3. Softmax features: phi(x) = exp(x) (original kernel)
    
    Time Complexity: O(n * d^2)
        - Computing K^T @ V: O(n * d^2)
        - Computing Q @ (K^T @ V): O(n * d^2)
    
    Space Complexity: O(n * d + d^2)
        - Linear in sequence length (no n^2 matrix)
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head.
        feature_map: Function to apply to Q and K.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        feature_map: str = "elu",
        eps: float = 1e-6,
        bias: bool = True
    ) -> None:
        """Initialize Linear Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            feature_map: Feature map to use - 'elu' or 'relu'.
            eps: Small constant for numerical stability.
            bias: Whether to include bias in linear projections.
        
        Raises:
            ValueError: If feature_map is not supported.
        """
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.eps = eps
        
        # Select feature map
        if feature_map == "elu":
            self.feature_map: Callable = elu_feature_map
        elif feature_map == "relu":
            self.feature_map = relu_feature_map
        else:
            raise ValueError(f"feature_map must be 'elu' or 'relu', got {feature_map}")
        
        self.feature_map_name = feature_map
        
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
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional mask (limited support in linear attention due to
                the associativity requirement).
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Approximate attention weights (computed for
                visualization, not used in actual computation) of shape
                (batch_size, num_heads, seq_len_q, seq_len_k).
        
        Note:
            The attention_weights returned are computed post-hoc for
            visualization purposes. The actual linear attention computation
            does not materialize the full n x n attention matrix.
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map to queries and keys
        # Shape: (batch, num_heads, seq_len, d_k)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Apply mask by zeroing out masked positions in K
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Expand mask for d_k dimension
            k = k.masked_fill(mask.bool().unsqueeze(-1), 0.0)
        
        # Linear attention computation
        # Instead of: softmax(QK^T) @ V which is O(n^2 * d)
        # We compute: Q @ (K^T @ V) which is O(n * d^2)
        
        # Compute K^T @ V: (batch, num_heads, d_k, d_k)
        kv = torch.einsum("bhnd,bhnv->bhdv", k, v)
        
        # Compute Q @ (K^T @ V): (batch, num_heads, seq_len_q, d_k)
        numerator = torch.einsum("bhqd,bhdv->bhqv", q, kv)
        
        # Compute normalization: Q @ K^T @ 1 = Q @ sum(K, dim=seq)
        # sum(K, dim=seq): (batch, num_heads, d_k)
        k_sum = k.sum(dim=2)
        # Q @ k_sum: (batch, num_heads, seq_len_q)
        denominator = torch.einsum("bhqd,bhd->bhq", q, k_sum)
        
        # Normalize
        denominator = denominator.unsqueeze(-1).clamp(min=self.eps)
        attended = numerator / denominator
        
        # Apply dropout (note: applied to output, not attention weights)
        attended = self.dropout(attended)
        
        # Reshape: (batch, seq_len_q, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.w_o(attended)
        
        # Compute approximate attention weights for visualization
        # This materializes the full matrix, so only use for small sequences
        with torch.no_grad():
            # (batch, num_heads, seq_len_q, seq_len_k)
            approx_weights = torch.einsum("bhqd,bhkd->bhqk", q, k)
            approx_weights = approx_weights / (approx_weights.sum(dim=-1, keepdim=True) + self.eps)
        
        return output, approx_weights
    
    def forward_causal(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute causal linear attention using RNN-style recurrence.
        
        This method demonstrates the key insight that linear attention can
        be computed autoregressively in O(1) time per step using a hidden
        state, making it equivalent to an RNN.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model).
            key: Key tensor of shape (batch_size, seq_len, d_model).
            value: Value tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            output: Attended values of shape (batch_size, seq_len, d_model).
            attention_weights: Empty tensor (not computed in causal mode).
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Initialize hidden states
        # S: (batch, num_heads, d_k, d_v) - cumulative sum of outer products
        # Z: (batch, num_heads, d_k) - cumulative sum of keys
        s = torch.zeros(batch_size, self.num_heads, self.d_k, self.d_k, 
                       device=query.device, dtype=query.dtype)
        z = torch.zeros(batch_size, self.num_heads, self.d_k,
                       device=query.device, dtype=query.dtype)
        
        outputs = []
        
        # Process each position sequentially
        for t in range(seq_len):
            # Get current position
            q_t = q[:, :, t, :]  # (batch, num_heads, d_k)
            k_t = k[:, :, t, :]  # (batch, num_heads, d_k)
            v_t = v[:, :, t, :]  # (batch, num_heads, d_k)
            
            # Update hidden state: S += k_t @ v_t^T
            s = s + torch.einsum("bhd,bhv->bhdv", k_t, v_t)
            z = z + k_t
            
            # Compute output: q_t @ S / (q_t @ z)
            num = torch.einsum("bhd,bhdv->bhv", q_t, s)
            den = torch.einsum("bhd,bhd->bh", q_t, z).unsqueeze(-1).clamp(min=self.eps)
            out_t = num / den
            
            outputs.append(out_t)
        
        # Stack outputs: (batch, num_heads, seq_len, d_k)
        attended = torch.stack(outputs, dim=2)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.d_model)
        output = self.w_o(attended)
        
        # Return empty attention weights (not computed in causal mode)
        dummy_weights = torch.zeros(batch_size, self.num_heads, seq_len, seq_len,
                                   device=query.device)
        
        return output, dummy_weights
    
    @property
    def complexity(self) -> str:
        """Return time complexity string.
        
        Returns:
            Time complexity as "O(n * d^2)".
        """
        return "O(n * d^2)"
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"feature_map={self.feature_map_name}, eps={self.eps}"
        )
