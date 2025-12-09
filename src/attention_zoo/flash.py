"""Flash Attention conceptual implementation.

Note: This is a conceptual/educational implementation demonstrating the
FlashAttention algorithm. The actual FlashAttention requires custom CUDA
kernels for the performance benefits. For production use, see:
- flash-attn package: https://github.com/Dao-AILab/flash-attention
- PyTorch 2.0+ scaled_dot_product_attention with flash attention backend
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttention


class FlashAttention(BaseAttention):
    """Flash Attention conceptual implementation.
    
    Implements the algorithm from "FlashAttention: Fast and Memory-Efficient 
    Exact Attention with IO-Awareness" (Dao et al., 2022):
    https://arxiv.org/abs/2205.14135
    
    FlashAttention achieves significant speedup and memory savings by:
    1. Tiling: Processing the attention matrix in blocks
    2. Recomputation: Trading compute for memory by not storing attention
    3. IO-Awareness: Minimizing HBM (GPU memory) reads/writes
    
    Key Insight: Standard attention is memory-bound, not compute-bound.
    The bottleneck is reading/writing the O(n^2) attention matrix to HBM.
    
    Algorithm Overview:
        1. Divide Q, K, V into blocks that fit in SRAM (fast on-chip memory)
        2. For each block of Q:
           - Load block of Q to SRAM
           - For each block of K, V:
             - Load blocks of K, V to SRAM
             - Compute attention for this block pair
             - Update running softmax using numerically stable algorithm
             - Accumulate output in SRAM
           - Write output block to HBM
    
    The key challenge is computing softmax across blocks, solved using
    the "online softmax" trick that maintains running max and sum.
    
    Time Complexity: O(n^2 * d)
        - Same as standard attention (exact computation)
    
    Space Complexity: O(n)
        - Does NOT store the n^2 attention matrix
        - Only O(block_size^2) memory for intermediate results
    
    Note: This implementation is for educational purposes. The actual
    performance benefits require custom CUDA kernels that exploit the
    memory hierarchy. In practice, use torch.nn.functional.scaled_dot_product_attention
    with enable_flash=True (PyTorch 2.0+) or the flash-attn package.
    
    Attributes:
        num_heads: Number of attention heads.
        d_k: Dimension per head.
        block_size_q: Block size for queries.
        block_size_kv: Block size for keys/values.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        block_size_q: int = 64,
        block_size_kv: int = 64,
        bias: bool = True
    ) -> None:
        """Initialize Flash Attention.
        
        Args:
            d_model: The dimensionality of the model.
            num_heads: Number of attention heads.
            dropout: Dropout probability (applied after softmax).
            block_size_q: Block size for tiling queries.
            block_size_kv: Block size for tiling keys/values.
            bias: Whether to include bias in linear projections.
        """
        super().__init__(d_model, dropout)
        
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        
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
    
    def _flash_attention_block(
        self,
        q_block: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute attention for a block of queries using online softmax.
        
        This demonstrates the core FlashAttention algorithm using the
        online softmax trick to process K, V in blocks.
        
        Args:
            q_block: Query block of shape (batch, num_heads, block_q, d_k).
            k: Full keys of shape (batch, num_heads, seq_len_k, d_k).
            v: Full values of shape (batch, num_heads, seq_len_k, d_k).
            mask: Optional mask.
        
        Returns:
            output: Block output of shape (batch, num_heads, block_q, d_k).
            max_scores: Running max for numerical stability.
            sum_exp: Running sum of exponentials.
        """
        batch_size = q_block.size(0)
        block_size_q = q_block.size(2)
        seq_len_k = k.size(2)
        
        # Initialize accumulators
        output = torch.zeros_like(q_block)
        max_scores = torch.full(
            (batch_size, self.num_heads, block_size_q, 1),
            float("-inf"),
            device=q_block.device,
            dtype=q_block.dtype
        )
        sum_exp = torch.zeros(
            (batch_size, self.num_heads, block_size_q, 1),
            device=q_block.device,
            dtype=q_block.dtype
        )
        
        # Process K, V in blocks
        num_kv_blocks = (seq_len_k + self.block_size_kv - 1) // self.block_size_kv
        
        for kv_block_idx in range(num_kv_blocks):
            kv_start = kv_block_idx * self.block_size_kv
            kv_end = min(kv_start + self.block_size_kv, seq_len_k)
            
            # Get K, V blocks
            k_block = k[:, :, kv_start:kv_end, :]
            v_block = v[:, :, kv_start:kv_end, :]
            
            # Compute scores for this block
            # (batch, num_heads, block_q, block_kv)
            scores_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
            
            # Apply mask if provided
            if mask is not None:
                mask_block = mask[:, :, :, kv_start:kv_end]
                scores_block = scores_block.masked_fill(mask_block.bool(), float("-inf"))
            
            # Online softmax update
            # Step 1: Compute new max
            block_max = scores_block.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_scores, block_max)
            
            # Step 2: Rescale previous sum and output
            exp_diff_old = torch.exp(max_scores - new_max)
            sum_exp = sum_exp * exp_diff_old
            output = output * exp_diff_old
            
            # Step 3: Compute exponentials for new block
            exp_scores = torch.exp(scores_block - new_max)
            
            # Step 4: Update sum and output
            sum_exp = sum_exp + exp_scores.sum(dim=-1, keepdim=True)
            output = output + torch.matmul(exp_scores, v_block)
            
            # Step 5: Update max
            max_scores = new_max
        
        # Normalize output
        output = output / sum_exp.clamp(min=1e-6)
        
        return output, max_scores, sum_exp
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Flash Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional attention mask.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Empty tensor (not materialized in Flash Attention).
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
        
        # Expand mask if needed
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        # Process Q in blocks
        num_q_blocks = (seq_len_q + self.block_size_q - 1) // self.block_size_q
        outputs = []
        
        for q_block_idx in range(num_q_blocks):
            q_start = q_block_idx * self.block_size_q
            q_end = min(q_start + self.block_size_q, seq_len_q)
            
            # Get Q block
            q_block = q[:, :, q_start:q_end, :]
            
            # Get corresponding mask block if mask is 2D (seq_q x seq_k)
            mask_block = None
            if mask is not None:
                if mask.size(2) > 1:
                    mask_block = mask[:, :, q_start:q_end, :]
                else:
                    mask_block = mask
            
            # Compute attention for this Q block
            out_block, _, _ = self._flash_attention_block(q_block, k, v, mask_block)
            outputs.append(out_block)
        
        # Concatenate outputs
        attended = torch.cat(outputs, dim=2)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        # Reshape: (batch, seq_len_q, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.w_o(attended)
        
        # Return empty attention weights (not materialized in Flash Attention)
        dummy_weights = torch.zeros(
            batch_size, self.num_heads, seq_len_q, seq_len_k,
            device=query.device
        )
        
        return output, dummy_weights
    
    def forward_with_pytorch_flash(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention using PyTorch's native Flash Attention.
        
        Uses torch.nn.functional.scaled_dot_product_attention with Flash
        Attention backend (available in PyTorch 2.0+).
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional attention mask.
        
        Returns:
            output: Attended values of shape (batch_size, seq_len_q, d_model).
            attention_weights: Empty tensor (not available with Flash).
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
        
        # Use PyTorch's SDPA with flash attention
        # Note: Requires PyTorch 2.0+
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else ~mask.bool(),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape: (batch, seq_len_q, d_model)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        output = self.w_o(attended)
        
        # Return empty attention weights
        dummy_weights = torch.zeros(
            batch_size, self.num_heads, seq_len_q, seq_len_k,
            device=query.device
        )
        
        return output, dummy_weights
    
    @property
    def complexity(self) -> str:
        """Return time complexity string.
        
        Returns:
            Time complexity as "O(n^2 * d)" with note about memory.
        """
        return "O(n^2 * d) time, O(n) memory"
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"block_size_q={self.block_size_q}, block_size_kv={self.block_size_kv}"
        )
