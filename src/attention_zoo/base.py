"""Abstract base class for attention mechanisms."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseAttention(ABC, nn.Module):
    """Abstract base class for all attention mechanisms.
    
    This class defines the interface that all attention mechanism implementations
    must follow. It provides common functionality such as dropout and defines
    the expected input/output signatures.
    
    Attributes:
        d_model: The dimensionality of the model (embedding dimension).
        dropout: Dropout layer for regularization.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        """Initialize the base attention mechanism.
        
        Args:
            d_model: The dimensionality of the model.
            dropout: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention output.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_v, d_model).
                Note: seq_len_k must equal seq_len_v.
            mask: Optional attention mask. Shape depends on implementation.
                Typically (batch_size, 1, 1, seq_len_k) or 
                (batch_size, 1, seq_len_q, seq_len_k).
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Attention distribution of shape 
                (batch_size, [num_heads,] seq_len_q, seq_len_k).
        """
        pass
    
    @property
    @abstractmethod
    def complexity(self) -> str:
        """Return the time complexity as a string.
        
        Returns:
            A string describing the time complexity, e.g., "O(n^2 * d)".
        """
        pass
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module.
        
        Returns:
            String containing d_model and dropout rate.
        """
        return f"d_model={self.d_model}, dropout={self.dropout.p}"
