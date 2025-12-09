"""Benchmark script for comparing attention mechanism performance."""

import time
from typing import Dict, List, Tuple

import torch
import matplotlib.pyplot as plt

from attention_zoo import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    CausalAttention,
    SparseAttention,
    LinearAttention,
    FlashAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)


def benchmark_forward(
    attention_module: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cpu"
) -> Tuple[float, float]:
    """Benchmark forward pass of an attention module.
    
    Args:
        attention_module: The attention module to benchmark.
        batch_size: Batch size.
        seq_len: Sequence length.
        d_model: Model dimension.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.
        device: Device to run on.
    
    Returns:
        Tuple of (mean_time_ms, std_time_ms).
    """
    attention_module = attention_module.to(device)
    attention_module.eval()
    
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = attention_module(q, k, v)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = attention_module(q, k, v)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def measure_memory(
    attention_module: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: str = "cuda"
) -> float:
    """Measure peak memory usage during forward pass.
    
    Args:
        attention_module: The attention module to benchmark.
        batch_size: Batch size.
        seq_len: Sequence length.
        d_model: Model dimension.
        device: Device (must be cuda for memory measurement).
    
    Returns:
        Peak memory usage in MB.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    
    attention_module = attention_module.to(device)
    attention_module.eval()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)
    
    with torch.no_grad():
        _ = attention_module(q, k, v)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    torch.cuda.empty_cache()
    
    return peak_memory


def run_benchmarks(
    sequence_lengths: List[int] = [128, 256, 512, 1024, 2048],
    d_model: int = 256,
    num_heads: int = 8,
    batch_size: int = 4,
    device: str = "cpu"
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Run benchmarks for all attention mechanisms.
    
    Args:
        sequence_lengths: List of sequence lengths to benchmark.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        batch_size: Batch size.
        device: Device to run on.
    
    Returns:
        Dictionary mapping mechanism name to {seq_len: (mean_ms, std_ms)}.
    """
    mechanisms = {
        "Scaled Dot-Product": lambda: ScaledDotProductAttention(d_model=d_model),
        "Multi-Head": lambda: MultiHeadAttention(d_model=d_model, num_heads=num_heads),
        "Causal": lambda: CausalAttention(d_model=d_model, num_heads=num_heads),
        "Sparse": lambda: SparseAttention(d_model=d_model, num_heads=num_heads, block_size=64),
        "Linear": lambda: LinearAttention(d_model=d_model, num_heads=num_heads),
        "Flash": lambda: FlashAttention(d_model=d_model, num_heads=num_heads),
        "Multi-Query": lambda: MultiQueryAttention(d_model=d_model, num_heads=num_heads),
        "Grouped-Query": lambda: GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=4),
        "Sliding Window": lambda: SlidingWindowAttention(d_model=d_model, num_heads=num_heads, window_size=128),
    }
    
    results: Dict[str, Dict[int, Tuple[float, float]]] = {}
    
    for name, create_fn in mechanisms.items():
        print(f"Benchmarking {name}...")
        results[name] = {}
        
        for seq_len in sequence_lengths:
            try:
                module = create_fn()
                mean_time, std_time = benchmark_forward(
                    module, batch_size, seq_len, d_model, device=device
                )
                results[name][seq_len] = (mean_time, std_time)
                print(f"  seq_len={seq_len}: {mean_time:.2f} +/- {std_time:.2f} ms")
            except Exception as e:
                print(f"  seq_len={seq_len}: Failed - {e}")
                results[name][seq_len] = (float("nan"), float("nan"))
    
    return results


def plot_results(
    results: Dict[str, Dict[int, Tuple[float, float]]],
    save_path: str = "figures/benchmark_results.png"
) -> None:
    """Plot benchmark results.
    
    Args:
        results: Benchmark results from run_benchmarks.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    
    for i, (name, data) in enumerate(results.items()):
        seq_lens = sorted(data.keys())
        means = [data[sl][0] for sl in seq_lens]
        stds = [data[sl][1] for sl in seq_lens]
        
        ax.errorbar(
            seq_lens, means, yerr=stds,
            label=name, marker="o", capsize=3,
            color=colors[i % len(colors)]
        )
    
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Forward Pass Time (ms)", fontsize=12)
    ax.set_title("Attention Mechanism Performance Comparison", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved benchmark plot to {save_path}")
    plt.close()


def print_summary_table(results: Dict[str, Dict[int, Tuple[float, float]]]) -> None:
    """Print a summary table of benchmark results.
    
    Args:
        results: Benchmark results from run_benchmarks.
    """
    seq_lens = sorted(list(results.values())[0].keys())
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (Forward Pass Time in ms)")
    print("=" * 80)
    
    # Header
    header = f"{'Mechanism':<20}"
    for sl in seq_lens:
        header += f" {sl:>10}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for name, data in results.items():
        row = f"{name:<20}"
        for sl in seq_lens:
            mean, _ = data.get(sl, (float("nan"), 0))
            row += f" {mean:>10.2f}"
        print(row)
    
    print("=" * 80)


def main():
    """Run benchmark suite."""
    print("Attention Mechanisms Benchmark")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Run benchmarks
    results = run_benchmarks(
        sequence_lengths=[128, 256, 512, 1024, 2048],
        d_model=256,
        num_heads=8,
        batch_size=4,
        device=device
    )
    
    # Print summary
    print_summary_table(results)
    
    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()
