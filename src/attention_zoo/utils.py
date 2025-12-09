"""Utility functions for attention mechanisms."""

import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[torch.nn.Dropout] = None,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.
    
    Implements the attention mechanism from "Attention Is All You Need"
    (Vaswani et al., 2017):
    
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        query: Query tensor of shape (..., seq_len_q, d_k).
        key: Key tensor of shape (..., seq_len_k, d_k).
        value: Value tensor of shape (..., seq_len_k, d_v).
        mask: Optional mask tensor. Positions with True/1 are masked (ignored).
            Shape should be broadcastable to (..., seq_len_q, seq_len_k).
        dropout: Optional dropout layer to apply to attention weights.
        scale: Optional scaling factor. Defaults to 1/sqrt(d_k).
    
    Returns:
        output: Attended values of shape (..., seq_len_q, d_v).
        attention_weights: Attention probabilities of shape (..., seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    
    # Compute attention scores: (batch, ..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), float("-inf"))
    
    # Compute attention probabilities
    attention_weights = F.softmax(scores, dim=-1)
    
    # Handle NaN from all-masked positions
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Compute output: (batch, ..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask.
    
    The mask prevents positions from attending to subsequent positions,
    which is essential for autoregressive generation.
    
    Args:
        seq_len: Sequence length.
        device: Device to create the mask on.
    
    Returns:
        Causal mask of shape (seq_len, seq_len) where True indicates
        positions that should be masked (not attended to).
    """
    # Create upper triangular matrix (True for positions to mask)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None
) -> torch.Tensor:
    """Create a padding mask from sequence lengths.
    
    Args:
        lengths: Tensor of sequence lengths of shape (batch_size,).
        max_len: Maximum sequence length. If None, uses max of lengths.
    
    Returns:
        Padding mask of shape (batch_size, max_len) where True indicates
        padding positions that should be masked.
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.size(0)
    # Create position indices: (1, max_len)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    # Create mask: True where position >= length
    mask = positions >= lengths.unsqueeze(1)
    
    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create a sliding window attention mask.
    
    Each position can only attend to positions within the window.
    
    Args:
        seq_len: Sequence length.
        window_size: Size of the attention window (one-sided).
        device: Device to create the mask on.
    
    Returns:
        Mask of shape (seq_len, seq_len) where True indicates positions
        that should be masked (outside the window).
    """
    # Create position difference matrix
    positions = torch.arange(seq_len, device=device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    
    # Mask positions outside the window
    mask = torch.abs(diff) > window_size
    
    return mask


def create_sparse_attention_mask(
    seq_len: int,
    block_size: int,
    num_global_tokens: int = 0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create a sparse attention mask with local and strided patterns.
    
    Implements a simplified version of the Sparse Transformer pattern
    (Child et al., 2019) with local attention blocks and optional
    global tokens.
    
    Args:
        seq_len: Sequence length.
        block_size: Size of local attention blocks.
        num_global_tokens: Number of tokens that attend to all positions.
        device: Device to create the mask on.
    
    Returns:
        Mask of shape (seq_len, seq_len) where True indicates positions
        that should be masked.
    """
    # Start with all positions masked
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    
    # Add local attention blocks
    for i in range(seq_len):
        block_start = (i // block_size) * block_size
        block_end = min(block_start + block_size, seq_len)
        mask[i, block_start:block_end] = False
    
    # Add strided attention (attend to every block_size-th position)
    for i in range(seq_len):
        for j in range(0, seq_len, block_size):
            mask[i, j] = False
    
    # Global tokens attend to and are attended by all
    if num_global_tokens > 0:
        mask[:num_global_tokens, :] = False
        mask[:, :num_global_tokens] = False
    
    return mask


def visualize_attention(
    attention_weights: torch.Tensor,
    query_labels: Optional[list] = None,
    key_labels: Optional[list] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor. Can be:
            - 2D: (seq_len_q, seq_len_k)
            - 3D: (num_heads, seq_len_q, seq_len_k) - will show all heads
            - 4D: (batch, num_heads, seq_len_q, seq_len_k) - uses first batch
        query_labels: Optional labels for query positions.
        key_labels: Optional labels for key positions.
        title: Title for the plot.
        figsize: Figure size as (width, height).
        cmap: Colormap for the heatmap.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Handle different input shapes
    if attention_weights.ndim == 4:
        # Use first batch
        attention_weights = attention_weights[0]
    
    if attention_weights.ndim == 3:
        # Multiple heads
        num_heads = attention_weights.shape[0]
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_heads == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for head_idx in range(num_heads):
            row, col = head_idx // cols, head_idx % cols
            ax = axes[row, col]
            
            sns.heatmap(
                attention_weights[head_idx],
                ax=ax,
                cmap=cmap,
                xticklabels=key_labels if key_labels else False,
                yticklabels=query_labels if query_labels else False,
                cbar=True,
                square=True
            )
            ax.set_title(f"Head {head_idx}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
        
        # Hide unused subplots
        for idx in range(num_heads, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
    
    else:
        # Single attention map (2D)
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attention_weights,
            ax=ax,
            cmap=cmap,
            xticklabels=key_labels if key_labels else False,
            yticklabels=query_labels if query_labels else False,
            cbar=True,
            square=True
        )
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage(tensor: torch.Tensor) -> str:
    """Get memory usage of a tensor in human-readable format.
    
    Args:
        tensor: PyTorch tensor.
    
    Returns:
        String describing memory usage (e.g., "256.00 MB").
    """
    bytes_used = tensor.element_size() * tensor.numel()
    
    if bytes_used >= 1024 ** 3:
        return f"{bytes_used / (1024 ** 3):.2f} GB"
    elif bytes_used >= 1024 ** 2:
        return f"{bytes_used / (1024 ** 2):.2f} MB"
    elif bytes_used >= 1024:
        return f"{bytes_used / 1024:.2f} KB"
    else:
        return f"{bytes_used} B"
