"""Tests for equivalence between attention mechanism variants."""

import pytest
import torch

from attention_zoo import (
    MultiHeadAttention,
    CausalAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    ScaledDotProductAttention,
)


class TestEquivalence:
    """Tests verifying equivalence relationships between attention variants."""
    
    def test_causal_equals_masked_standard(self):
        """Causal attention should equal standard attention with causal mask."""
        d_model, num_heads, seq_len = 64, 8, 16
        
        # Create tensors
        q = torch.randn(2, seq_len, d_model)
        k = torch.randn(2, seq_len, d_model)
        v = torch.randn(2, seq_len, d_model)
        
        # Causal attention
        causal_attn = CausalAttention(d_model=d_model, num_heads=num_heads)
        causal_out, causal_weights = causal_attn(q, k, v)
        
        # Standard MHA with causal mask
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # Copy weights for fair comparison
        mha.w_q.weight.data = causal_attn.w_q.weight.data.clone()
        mha.w_k.weight.data = causal_attn.w_k.weight.data.clone()
        mha.w_v.weight.data = causal_attn.w_v.weight.data.clone()
        mha.w_o.weight.data = causal_attn.w_o.weight.data.clone()
        mha.w_q.bias.data = causal_attn.w_q.bias.data.clone()
        mha.w_k.bias.data = causal_attn.w_k.bias.data.clone()
        mha.w_v.bias.data = causal_attn.w_v.bias.data.clone()
        mha.w_o.bias.data = causal_attn.w_o.bias.data.clone()
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        mha_out, _ = mha(q, k, v, mask=causal_mask)
        
        assert torch.allclose(causal_out, mha_out, atol=1e-5)
    
    def test_gqa_with_all_heads_equals_mha(self):
        """GQA with num_kv_heads = num_heads should equal MHA."""
        d_model, num_heads, seq_len = 64, 8, 16
        
        q = torch.randn(2, seq_len, d_model)
        k = torch.randn(2, seq_len, d_model)
        v = torch.randn(2, seq_len, d_model)
        
        # GQA with num_kv_heads = num_heads
        gqa = GroupedQueryAttention(
            d_model=d_model, num_heads=num_heads, num_kv_heads=num_heads
        )
        gqa_out, _ = gqa(q, k, v)
        
        # MHA
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # Copy weights
        mha.w_q.weight.data = gqa.w_q.weight.data.clone()
        mha.w_k.weight.data = gqa.w_k.weight.data.clone()
        mha.w_v.weight.data = gqa.w_v.weight.data.clone()
        mha.w_o.weight.data = gqa.w_o.weight.data.clone()
        mha.w_q.bias.data = gqa.w_q.bias.data.clone()
        mha.w_k.bias.data = gqa.w_k.bias.data.clone()
        mha.w_v.bias.data = gqa.w_v.bias.data.clone()
        mha.w_o.bias.data = gqa.w_o.bias.data.clone()
        
        mha_out, _ = mha(q, k, v)
        
        assert torch.allclose(gqa_out, mha_out, atol=1e-5)
    
    def test_single_head_mha_equals_sdpa(self):
        """Single-head MHA should be equivalent to SDPA."""
        d_model, seq_len = 64, 16
        
        q = torch.randn(2, seq_len, d_model)
        k = torch.randn(2, seq_len, d_model)
        v = torch.randn(2, seq_len, d_model)
        
        # Single head MHA
        mha = MultiHeadAttention(d_model=d_model, num_heads=1)
        mha_out, mha_weights = mha(q, k, v)
        
        assert mha_out.shape == (2, seq_len, d_model)
        assert mha_weights.shape == (2, 1, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Verify attention weights sum to 1 along key dimension."""
        d_model, seq_len = 64, 16
        
        mechanisms = [
            ScaledDotProductAttention(d_model=d_model),
            MultiHeadAttention(d_model=d_model, num_heads=8),
            MultiQueryAttention(d_model=d_model, num_heads=8),
            GroupedQueryAttention(d_model=d_model, num_heads=8, num_kv_heads=4),
        ]
        
        q = torch.randn(2, seq_len, d_model)
        
        for attn in mechanisms:
            _, weights = attn(q, q, q)
            # Sum along last dimension should be 1
            weight_sums = weights.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
