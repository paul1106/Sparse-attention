"""
Benchmark dense attention (PyTorch baseline).

Tests PyTorch's native scaled_dot_product_attention at various scales.
"""

import torch
import torch.nn.functional as F
import time
import os


def benchmark_dense_attention(Q, K, V, num_runs=100, warmup=10):
    """
    Benchmark dense attention.
    
    Args:
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        num_runs: number of benchmark runs
        warmup: number of warmup runs
    
    Returns:
        avg_time: average time in milliseconds
    """
    device = Q.device
    
    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize(device)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = F.scaled_dot_product_attention(Q, K, V)
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
    
    data = torch.load(filename)
    Q = data['Q'].cuda()
    K = data['K'].cuda()
    V = data['V'].cuda()
    
    return Q, K, V


def main():
    """Run dense attention benchmarks."""
    
    test_files = [
        ("1K", "test_data_1k.pt"),
        ("2K", "test_data_2k.pt"),
        ("4K", "test_data_4k.pt"),
        ("8K", "test_data_8k.pt"),
        ("16K", "test_data_16k.pt"),
        ("32K", "test_data_32k.pt"),
        ("64K", "test_data_64k.pt"),
        ("128K", "test_data_128k.pt"),
    ]
    
    print("=" * 80)
    print("Dense Attention Benchmark (PyTorch Baseline)")
    print("=" * 80)
    print()
    
    results = []
    
    for name, filename in test_files:
        print(f"Testing {name} tokens...")
        
        data = load_test_data(filename)
        if data is None:
            print(f"  ❌ File not found: {filename}")
            print(f"  Run generate_test_data.py first!\n")
            continue
        
        Q, K, V = data
        batch, num_heads, seq_len, head_dim = Q.shape
        
        print(f"  Shape: [{batch}, {num_heads}, {seq_len}, {head_dim}]")
        
        # Calculate memory
        qkv_memory = (Q.numel() + K.numel() + V.numel()) * 2 / (1024**3)  # GB
        attn_matrix_memory = batch * num_heads * seq_len * seq_len * 2 / (1024**3)  # GB
        
        print(f"  QKV Memory: {qkv_memory:.2f} GB")
        print(f"  Attention Matrix: {attn_matrix_memory:.2f} GB")
        
        try:
            # Benchmark
            avg_time, std_time = benchmark_dense_attention(Q, K, V)
            
            print(f"  ✅ Time: {avg_time:.3f} ± {std_time:.3f} ms")
            
            # Calculate throughput
            flops = 4 * batch * num_heads * seq_len * seq_len * head_dim  # 2 matmuls
            tflops = flops / (avg_time / 1000) / 1e12
            
            print(f"  Throughput: {tflops:.2f} TFLOPS")
            print()
            
            results.append({
                'name': name,
                'seq_len': seq_len,
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
    print(f"{'Config':<10} {'Seq Len':<10} {'Time (ms)':<15} {'TFLOPS':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<10} {r['seq_len']:<10} {r['time_ms']:>7.3f} ± {r['std_ms']:<5.3f} {r['tflops']:>8.2f}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
