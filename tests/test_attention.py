"""Unit tests for attention mechanisms."""

import pytest
import torch

from attention_zoo import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    CrossAttention,
    CausalAttention,
    SparseAttention,
    LinearAttention,
    FlashAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)


class TestScaledDotProductAttention:
    """Tests for ScaledDotProductAttention."""
    
    def test_output_shape(self):
        attn = ScaledDotProductAttention(d_model=64)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 10, 10)
    
    def test_gradient_flow(self):
        attn = ScaledDotProductAttention(d_model=64)
        q = torch.randn(2, 10, 64, requires_grad=True)
        k = torch.randn(2, 10, 64, requires_grad=True)
        v = torch.randn(2, 10, 64, requires_grad=True)
        
        output, _ = attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
    
    def test_complexity_property(self):
        attn = ScaledDotProductAttention(d_model=64)
        assert attn.complexity == "O(n^2 * d)"


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)
    
    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=63, num_heads=8)
    
    def test_mask_application(self):
        attn = MultiHeadAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        mask = torch.ones(2, 1, 1, 10).bool()
        mask[:, :, :, :5] = False
        
        _, weights = attn(q, k, v, mask=mask)
        
        assert weights[:, :, :, 5:].sum() == 0


class TestCrossAttention:
    """Tests for CrossAttention."""
    
    def test_different_sequence_lengths(self):
        attn = CrossAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 16, 64)
        v = torch.randn(2, 16, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 8, 64)
        assert weights.shape == (2, 8, 8, 16)


class TestCausalAttention:
    """Tests for CausalAttention."""
    
    def test_causal_masking(self):
        attn = CausalAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        _, weights = attn(q, k, v)
        
        # Check upper triangular is zero (causal mask applied)
        for i in range(10):
            for j in range(i + 1, 10):
                assert weights[:, :, i, j].sum() == 0


class TestSparseAttention:
    """Tests for SparseAttention."""
    
    def test_output_shape(self):
        attn = SparseAttention(d_model=64, num_heads=8, block_size=4)
        q = torch.randn(2, 16, 64)
        k = torch.randn(2, 16, 64)
        v = torch.randn(2, 16, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 16, 64)
    
    def test_patterns(self):
        for pattern in ["strided", "fixed", "combined"]:
            attn = SparseAttention(d_model=64, num_heads=8, pattern=pattern)
            q = torch.randn(2, 16, 64)
            output, _ = attn(q, q, q)
            assert output.shape == (2, 16, 64)


class TestLinearAttention:
    """Tests for LinearAttention."""
    
    def test_output_shape(self):
        attn = LinearAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, _ = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)
    
    def test_feature_maps(self):
        for fm in ["elu", "relu"]:
            attn = LinearAttention(d_model=64, num_heads=8, feature_map=fm)
            q = torch.randn(2, 10, 64)
            output, _ = attn(q, q, q)
            assert output.shape == (2, 10, 64)


class TestFlashAttention:
    """Tests for FlashAttention."""
    
    def test_output_shape(self):
        attn = FlashAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, _ = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)


class TestMultiQueryAttention:
    """Tests for MultiQueryAttention."""
    
    def test_output_shape(self):
        attn = MultiQueryAttention(d_model=64, num_heads=8)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)


class TestGroupedQueryAttention:
    """Tests for GroupedQueryAttention."""
    
    def test_output_shape(self):
        attn = GroupedQueryAttention(d_model=64, num_heads=8, num_kv_heads=4)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 8, 10, 10)
    
    def test_invalid_num_kv_heads(self):
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_model=64, num_heads=8, num_kv_heads=3)


class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""
    
    def test_output_shape(self):
        attn = SlidingWindowAttention(d_model=64, num_heads=8, window_size=4)
        q = torch.randn(2, 16, 64)
        k = torch.randn(2, 16, 64)
        v = torch.randn(2, 16, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (2, 16, 64)
    
    def test_window_masking(self):
        attn = SlidingWindowAttention(d_model=64, num_heads=8, window_size=2)
        q = torch.randn(2, 10, 64)
        
        _, weights = attn(q, q, q)
        
        # Positions more than window_size apart should have zero attention
        assert weights[:, :, 0, 5].sum() == 0
        assert weights[:, :, 0, 2].sum() != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
