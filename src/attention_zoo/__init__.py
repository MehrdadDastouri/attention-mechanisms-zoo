"""Attention Mechanisms Zoo - A collection of attention implementations."""

from .base import BaseAttention
from .scaled_dot_product import ScaledDotProductAttention
from .multi_head import MultiHeadAttention
from .cross_attention import CrossAttention
from .causal import CausalAttention
from .sparse import SparseAttention
from .linear import LinearAttention
from .flash import FlashAttention
from .multi_query import MultiQueryAttention
from .grouped_query import GroupedQueryAttention
from .sliding_window import SlidingWindowAttention
from .utils import (
    scaled_dot_product_attention,
    create_causal_mask,
    create_padding_mask,
    create_sliding_window_mask,
    create_sparse_attention_mask,
    visualize_attention,
    count_parameters,
    get_memory_usage,
)

__version__ = "0.1.0"

__all__ = [
    # Base class
    "BaseAttention",
    # Attention mechanisms
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "CrossAttention",
    "CausalAttention",
    "SparseAttention",
    "LinearAttention",
    "FlashAttention",
    "MultiQueryAttention",
    "GroupedQueryAttention",
    "SlidingWindowAttention",
    # Utilities
    "scaled_dot_product_attention",
    "create_causal_mask",
    "create_padding_mask",
    "create_sliding_window_mask",
    "create_sparse_attention_mask",
    "visualize_attention",
    "count_parameters",
    "get_memory_usage",
]
