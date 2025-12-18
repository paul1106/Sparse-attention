"""
Benchmark Sliding Window Attention (PyTorch implementation).

Tests PyTorch's sparse attention with sliding window pattern.
"""

import torch
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patterns import sliding_window_attention, measure_sparsity, sliding_window_mask


def benchmark_sliding_window_attention(Q, K, V, window_size, num_runs=100, warmup=10):
    """
    Benchmark sliding window attention.
    
    Args:
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        window_size: int
        num_runs: number of benchmark runs
        warmup: number of warmup runs
    
    Returns:
        avg_time: average time in milliseconds
    """
    device = Q.device
    
    # Warmup
    for _ in range(warmup):
        _ = sliding_window_attention(Q, K, V, window_size)
    torch.cuda.synchronize(device)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = sliding_window_attention(Q, K, V, window_size)
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_time


def load_test_data(filename):
    """Load Q, K, V from file."""
    if not os.path.exists(filename):
        return None
    
    data = torch.load(filename, weights_only=False)
    Q = data['Q'].cuda()
    K = data['K'].cuda()
    V = data['V'].cuda()
    
    return Q, K, V


def main():
    """Run sliding window attention benchmarks."""
    
    # Test configurations: (name, filename, window_size)
    test_configs = [
        ("1K", "test_data_1k.pt", 64),
        ("2K", "test_data_2k.pt", 128),
        ("4K", "test_data_4k.pt", 256),
        ("8K", "test_data_8k.pt", 512),
        ("16K", "test_data_16k.pt", 1024),
        ("32K", "test_data_32k.pt", 2048),
        ("64K", "test_data_64k.pt", 4096),
        ("128K", "test_data_128k.pt", 4096),
    ]
    
    print("=" * 80)
    print("Sliding Window Attention Benchmark (PyTorch)")
    print("=" * 80)
    print()
    
    results = []
    
    for name, filename, window_size in test_configs:
        print(f"Testing {name} tokens (window={window_size})...")
        
        data = load_test_data(filename)
        if data is None:
            print(f"  ❌ File not found: {filename}")
            print()
            continue
        
        Q, K, V = data
        batch, num_heads, seq_len, head_dim = Q.shape
        
        print(f"  Shape: [{batch}, {num_heads}, {seq_len}, {head_dim}]")
        
        # Calculate sparsity
        mask = sliding_window_mask(seq_len, window_size, device='cpu')
        sparsity = measure_sparsity(mask)
        print(f"  Sparsity: {sparsity:.2f}%")
        
        # Calculate memory
        qkv_memory = (Q.numel() + K.numel() + V.numel()) * 2 / (1024**3)  # GB
        
        # For sparse, only non-zero elements matter
        avg_nnz_per_row = min(2 * window_size + 1, seq_len)
        total_nnz = seq_len * avg_nnz_per_row
        sparse_attn_memory = batch * num_heads * total_nnz * 2 / (1024**3)  # GB
        
        print(f"  QKV Memory: {qkv_memory:.2f} GB")
        print(f"  Sparse Attention Matrix: {sparse_attn_memory:.2f} GB (vs {batch * num_heads * seq_len * seq_len * 2 / (1024**3):.2f} GB dense)")
        
        try:
            # Benchmark
            avg_time, std_time = benchmark_sliding_window_attention(Q, K, V, window_size)
            
            print(f"  ✅ Time: {avg_time:.3f} ± {std_time:.3f} ms")
            
            # Calculate throughput (only for non-zero elements)
            flops = 4 * batch * num_heads * total_nnz * head_dim  # sparse matmuls
            tflops = flops / (avg_time / 1000) / 1e12
            
            print(f"  Throughput: {tflops:.2f} TFLOPS (sparse)")
            print()
            
            results.append({
                'name': name,
                'seq_len': seq_len,
                'window_size': window_size,
                'sparsity': sparsity,
                'time_ms': avg_time,
                'std_ms': std_time,
                'tflops': tflops,
            })
            
        except RuntimeError as e:
            print(f"  ❌ Failed: {e}")
            print()
    
    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Config':<10} {'Seq Len':<10} {'Window':<10} {'Sparsity':<12} {'Time (ms)':<15} {'TFLOPS':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<10} {r['seq_len']:<10} {r['window_size']:<10} {r['sparsity']:>9.2f}% {r['time_ms']:>7.3f} ± {r['std_ms']:<5.3f} {r['tflops']:>8.2f}")
    
    print("=" * 80)
    
    # Calculate speedup vs dense baseline (hardcoded from previous run)
    dense_times = {
        1024: 0.140,
        2048: 0.480,
        4096: 1.742,
        8192: 6.823,
        16384: 27.141,
        32768: 108.601,
        65536: 441.216,
        131072: 1785.037,
    }
    
    print()
    print("=" * 80)
    print("Speedup vs Dense Baseline")
    print("=" * 80)
    print(f"{'Config':<10} {'Dense (ms)':<15} {'Sparse (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for r in results:
        seq_len = r['seq_len']
        if seq_len in dense_times:
            dense_time = dense_times[seq_len]
            sparse_time = r['time_ms']
            speedup = dense_time / sparse_time
            
            status = "✅" if speedup > 1.0 else "❌"
            print(f"{r['name']:<10} {dense_time:>13.3f} {sparse_time:>13.3f} {speedup:>8.2f}x {status}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
