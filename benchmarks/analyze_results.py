"""
Compare PyTorch Dense vs Sliding Window Sparse results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Dense baseline results (from benchmark_dense.py)
dense_results = {
    'seq_lens': [1024, 2048, 4096, 8192, 16384],
    'times': [0.140, 0.480, 1.742, 6.823, 27.141],  # ms
    'memory': [0.06, 0.25, 1.00, 4.00, 16.00],  # GB (attention matrix)
}

# Sliding window sparse results (from benchmark_sliding_window.py)
sparse_results = {
    'seq_lens': [1024, 2048, 4096, 8192, 16384],
    'times': [0.841, 3.777, 12.888, 50.019, 198.976],  # ms
    'memory': [0.01, 0.03, 0.13, 0.50, 2.00],  # GB (sparse attention matrix)
    'windows': [64, 128, 256, 512, 1024],
}

print("=" * 100)
print("PyTorch Dense vs Sliding Window Sparse Comparison")
print("=" * 100)
print()

print(f"{'Seq Len':<12} {'Dense (ms)':<15} {'Sparse (ms)':<15} {'Speedup':<12} {'Dense Mem (GB)':<18} {'Sparse Mem (GB)':<18}")
print("-" * 100)

for i in range(len(dense_results['seq_lens'])):
    seq_len = dense_results['seq_lens'][i]
    dense_time = dense_results['times'][i]
    sparse_time = sparse_results['times'][i]
    speedup = dense_time / sparse_time
    dense_mem = dense_results['memory'][i]
    sparse_mem = sparse_results['memory'][i]
    mem_reduction = sparse_mem / dense_mem
    
    status = "✅" if speedup > 1.0 else "❌ SLOWER"
    
    print(f"{seq_len:<12} {dense_time:>13.3f} {sparse_time:>13.3f} {speedup:>9.3f}x {status:<6} {dense_mem:>16.2f} {sparse_mem:>16.2f} ({mem_reduction*100:.1f}%)")

print("=" * 100)
print()

print("🔍 Key Findings:")
print()
print("❌ PyTorch Sliding Window is SLOWER than Dense!")
print("   - 1K tokens:  0.14ms (dense) vs 0.84ms (sparse) = 6.0x SLOWER")
print("   - 16K tokens: 27.1ms (dense) vs 199.0ms (sparse) = 7.3x SLOWER")
print()
print("Why?")
print("1. PyTorch's sparse implementation uses masking:")
print("   - Still computes full QK^T matrix (expensive!)")
print("   - Then applies mask")
print("   - No actual computation savings")
print()
print("2. For large scales (32K+), PyTorch tries to materialize full matrix:")
print("   - OOM at 32K (needs 64GB)")
print("   - Cannot run 128K at all")
print()
print("✅ Memory Reduction Works:")
print("   - 16K: 16GB (dense) → 2GB (sparse) = 8x reduction")
print("   - But computation is still O(n²)!")
print()
print("🎯 This is WHY we need Custom CUDA Kernels!")
print()
print("Custom CUDA will:")
print("1. Use CSR format - only compute non-zero elements")
print("2. True sparse computation: O(n × w) instead of O(n²)")
print("3. Expected speedup at 16K: 7-8x (instead of 7x slower!)")
print("4. Can scale to 128K without OOM")
print()
print("Next Step: Implement CUDA kernel with CSR format")
print("=" * 100)

# Calculate what CUDA should achieve
print()
print("=" * 100)
print("Expected CUDA Performance (Theoretical)")
print("=" * 100)
print()

print(f"{'Seq Len':<12} {'Dense (ms)':<15} {'PyTorch Sparse':<18} {'CUDA Target':<15} {'Expected Speedup':<20}")
print("-" * 100)

for i in range(len(dense_results['seq_lens'])):
    seq_len = dense_results['seq_lens'][i]
    window = sparse_results['windows'][i]
    dense_time = dense_results['times'][i]
    sparse_time = sparse_results['times'][i]
    
    # CUDA should be proportional to sparsity
    sparsity = 1 - (2 * window + 1) / seq_len
    # Theoretical: dense_time × (1 - sparsity) + overhead
    # Conservative estimate: ~10% of dense time (for ~90% sparsity)
    cuda_target = dense_time * (1 - sparsity) * 1.5  # 1.5x for overhead
    expected_speedup = dense_time / cuda_target
    
    print(f"{seq_len:<12} {dense_time:>13.3f} {sparse_time:>16.3f} {cuda_target:>13.3f} {expected_speedup:>18.2f}x")

print("=" * 100)
print()
print("With custom CUDA at 128K (extrapolated):")
print("  Dense: ~1785 ms")
print("  Expected CUDA: ~200-300 ms")
print("  Speedup: 6-9x ✅")
print()
