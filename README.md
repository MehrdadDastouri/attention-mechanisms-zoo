# Attention Mechanisms Zoo

A comprehensive collection of attention mechanism implementations from seminal research papers, with complete mathematical derivations, PyTorch implementations, and educational Jupyter notebooks.

## Overview

This repository provides production-ready implementations of 10 attention mechanisms, each with:
- Complete mathematical derivation
- Time and space complexity analysis
- Clean, type-hinted PyTorch code
- Visualization utilities
- Educational Jupyter notebooks

## Attention Mechanisms

| Mechanism | Paper | Time Complexity | Space Complexity | Use Case |
|-----------|-------|-----------------|------------------|----------|
| Scaled Dot-Product | Vaswani et al., 2017 | O(n²d) | O(n²) | Standard transformer attention |
| Multi-Head | Vaswani et al., 2017 | O(n²d) | O(n²h) | Parallel subspace attention |
| Cross-Attention | Vaswani et al., 2017 | O(nmd) | O(nm) | Encoder-decoder models |
| Causal (Masked) | Vaswani et al., 2017 | O(n²d) | O(n²) | Autoregressive generation |
| Sparse | Child et al., 2019 | O(n√n) | O(n√n) | Long sequences with patterns |
| Linear | Katharopoulos et al., 2020 | O(nd²) | O(nd) | Very long sequences |
| Flash | Dao et al., 2022 | O(n²d) | O(n) | Memory-efficient training |
| Multi-Query | Shazeer, 2019 | O(n²d) | O(n²) | Fast inference |
| Grouped-Query | Ainslie et al., 2023 | O(n²d) | O(n²) | Balance speed/quality |
| Sliding Window | Beltagy et al., 2020 | O(nwd) | O(nw) | Local context modeling |

Where: n = sequence length, d = model dimension, h = number of heads, m = encoder sequence length, w = window size

## Installation

```bash
git clone https://github.com/yourusername/attention-mechanisms-zoo.git
cd attention-mechanisms-zoo
pip install -e .
```

## Quick Start

```python
import torch
from attention_zoo import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    CausalAttention,
    LinearAttention,
)

# Initialize attention mechanism
attention = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

# Create sample input
batch_size, seq_len, d_model = 2, 128, 512
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, attention_weights = attention(query, key, value)
print(f"Output shape: {output.shape}")  # (2, 128, 512)
print(f"Attention weights shape: {attention_weights.shape}")  # (2, 8, 128, 128)
```

## Project Structure

```
attention-mechanisms-zoo/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── 01_scaled_dot_product_attention.ipynb
│   ├── 02_multi_head_attention.ipynb
│   ├── 03_cross_attention.ipynb
│   ├── 04_masked_causal_attention.ipynb
│   ├── 05_sparse_attention.ipynb
│   ├── 06_linear_attention.ipynb
│   ├── 07_flash_attention.ipynb
│   ├── 08_multi_query_attention.ipynb
│   ├── 09_grouped_query_attention.ipynb
│   └── 10_sliding_window_attention.ipynb
├── src/
│   └── attention_zoo/
│       ├── __init__.py
│       ├── base.py
│       ├── scaled_dot_product.py
│       ├── multi_head.py
│       ├── cross_attention.py
│       ├── causal.py
│       ├── sparse.py
│       ├── linear.py
│       ├── flash.py
│       ├── multi_query.py
│       ├── grouped_query.py
│       ├── sliding_window.py
│       └── utils.py
├── tests/
│   ├── test_attention.py
│   └── test_equivalence.py
├── benchmarks/
│   └── benchmark_attention.py
└── figures/
    └── .gitkeep
```

## When to Use Which Attention

### Scaled Dot-Product Attention
- **Use when**: Building standard transformer components
- **Avoid when**: Sequence length exceeds available memory

### Multi-Head Attention
- **Use when**: Need to capture diverse relationships in different subspaces
- **Avoid when**: Computational budget is extremely limited

### Cross-Attention
- **Use when**: Encoder-decoder architectures (translation, image captioning)
- **Avoid when**: Self-attention tasks

### Causal (Masked) Attention
- **Use when**: Autoregressive generation (GPT-style models)
- **Avoid when**: Bidirectional context is needed

### Sparse Attention
- **Use when**: Very long sequences with structured patterns
- **Avoid when**: Dense attention patterns are required

### Linear Attention
- **Use when**: Extremely long sequences where O(n^2) is prohibitive
- **Avoid when**: Precise softmax attention behavior is required

### Flash Attention
- **Use when**: Training large models with GPU memory constraints
- **Avoid when**: Custom attention patterns are needed

### Multi-Query Attention
- **Use when**: Inference speed is critical (large batch decoding)
- **Avoid when**: Training from scratch with quality priority

### Grouped-Query Attention
- **Use when**: Balance between MHA quality and MQA speed
- **Avoid when**: Either extreme (quality or speed) is acceptable

### Sliding Window Attention
- **Use when**: Local context is sufficient (many NLP tasks)
- **Avoid when**: Global context is essential

## Notebooks

Each notebook provides:
1. **Paper Reference**: Full citation with arXiv/conference link
2. **Mathematical Derivation**: Step-by-step formulas in LaTeX
3. **Complexity Analysis**: Big-O time and space analysis
4. **Implementation**: Clean PyTorch code with type hints
5. **Visualization**: Attention pattern heatmaps
6. **Comparison**: When to use this variant

## Benchmarks

Run benchmarks to compare performance:

```bash
python benchmarks/benchmark_attention.py
```

This will generate timing and memory comparisons across different sequence lengths.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers." [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)

3. Katharopoulos, A., et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

4. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

5. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

6. Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

7. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer." [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{attention_mechanisms_zoo,
  title={Attention Mechanisms Zoo},
  author=Mehrdaddastouri,
  year={2024},
  url={https://github.com/mehrdaddastouri/attention-mechanisms-zoo}
}
```
