"""
Test sliding window attention pattern.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patterns import (
    sliding_window_mask,
    sliding_window_attention,
    sliding_window_causal_mask,
    attention_dense,
    to_csr,
    measure_sparsity,
    visualize_sparsity_pattern,
    print_csr_stats
)


def test_sliding_window_pattern():
    """Test sliding window pattern generation."""
    print("=" * 60)
    print("Test 1: Sliding Window Pattern")
    print("=" * 60)
    
    seq_len = 16
    window_size = 3
    
    mask = sliding_window_mask(seq_len, window_size, device='cpu')
    
    print(f"Sequence Length: {seq_len}")
    print(f"Window Size: {window_size}")
    print(f"Sparsity: {measure_sparsity(mask):.2f}%")
    print()
    print("Mask Pattern (1 = attend, 0 = ignore):")
    print(mask.int())
    print()


def test_causal_sliding_window():
    """Test causal sliding window pattern."""
    print("=" * 60)
    print("Test 2: Causal Sliding Window Pattern")
    print("=" * 60)
    
    seq_len = 16
    window_size = 3
    
    mask = sliding_window_causal_mask(seq_len, window_size, device='cpu')
    
    print(f"Sequence Length: {seq_len}")
    print(f"Window Size: {window_size}")
    print(f"Sparsity: {measure_sparsity(mask):.2f}%")
    print()
    print("Mask Pattern (1 = attend, 0 = ignore):")
    print(mask.int())
    print()


def test_csr_conversion():
    """Test CSR format conversion."""
    print("=" * 60)
    print("Test 3: CSR Format Conversion")
    print("=" * 60)
    
    seq_len = 8
    window_size = 2
    
    mask = sliding_window_mask(seq_len, window_size, device='cpu')
    row_ptr, col_indices, nnz = to_csr(mask)
    
    print(f"Sequence Length: {seq_len}")
    print(f"Window Size: {window_size}")
    print()
    
    print_csr_stats(row_ptr, col_indices)
    print()
    
    print("CSR Representation:")
    print(f"  row_ptr: {row_ptr.tolist()}")
    print(f"  col_indices: {col_indices.tolist()}")
    print()


def test_attention_correctness():
    """Test that sliding window attention produces correct output."""
    print("=" * 60)
    print("Test 4: Attention Correctness")
    print("=" * 60)
    
    batch = 1
    num_heads = 2
    seq_len = 32
    head_dim = 64
    window_size = 8
    
    # Generate random Q, K, V
    Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
    K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
    V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
    
    # Compute with sliding window
    mask = sliding_window_mask(seq_len, window_size, device='cpu')
    output_sparse = sliding_window_attention(Q.cpu(), K.cpu(), V.cpu(), window_size)
    
    # Compute dense attention with same mask
    output_dense = attention_dense(Q.cpu(), K.cpu(), V.cpu(), mask=mask)
    
    # Compare
    max_diff = (output_sparse - output_dense).abs().max().item()
    mean_diff = (output_sparse - output_dense).abs().mean().item()
    
    print(f"Shape: [{batch}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"Window Size: {window_size}")
    print(f"Max Diff: {max_diff:.6f}")
    print(f"Mean Diff: {mean_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ Correctness test PASSED")
    else:
        print("❌ Correctness test FAILED")
    print()


def test_large_scale_sparsity():
    """Test sparsity at large scales."""
    print("=" * 60)
    print("Test 5: Large-Scale Sparsity Analysis")
    print("=" * 60)
    
    configs = [
        (1024, 64),
        (4096, 256),
        (8192, 512),
        (16384, 1024),
        (32768, 2048),
        (65536, 4096),
        (131072, 4096),
    ]
    
    print(f"{'Seq Len':<10} {'Window':<10} {'Sparsity':<12} {'NNZ':<15} {'Memory Reduction':<20}")
    print("-" * 70)
    
    for seq_len, window_size in configs:
        # Calculate without materializing full mask
        # Each row has at most (2 * window_size + 1) non-zero elements
        # Edge rows have fewer
        
        # For simplicity, approximate:
        avg_nnz_per_row = min(2 * window_size + 1, seq_len)
        total_nnz = seq_len * avg_nnz_per_row
        total_elements = seq_len * seq_len
        sparsity = (1 - total_nnz / total_elements) * 100
        
        # Memory calculation
        csr_memory = (seq_len + 1 + total_nnz) * 4  # int32
        dense_memory = total_elements * 2  # fp16
        memory_reduction = (1 - csr_memory / dense_memory) * 100
        
        print(f"{seq_len:<10} {window_size:<10} {sparsity:>10.2f}% {total_nnz:>13,} {memory_reduction:>18.1f}%")
    
    print()


def visualize_patterns():
    """Visualize different sliding window patterns."""
    print("=" * 60)
    print("Test 6: Pattern Visualization")
    print("=" * 60)
    
    seq_len = 64
    window_sizes = [2, 4, 8, 16]
    
    for window_size in window_sizes:
        mask = sliding_window_mask(seq_len, window_size, device='cpu')
        filename = f"results/sliding_window_{seq_len}_w{window_size}.png"
        
        os.makedirs("results", exist_ok=True)
        visualize_sparsity_pattern(
            mask,
            title=f"Sliding Window (seq={seq_len}, w={window_size})",
            save_path=filename
        )
        
        print(f"Window {window_size}: {measure_sparsity(mask):.2f}% sparse -> {filename}")
    
    # Also visualize causal version
    window_size = 8
    mask = sliding_window_causal_mask(seq_len, window_size, device='cpu')
    filename = f"results/sliding_window_causal_{seq_len}_w{window_size}.png"
    visualize_sparsity_pattern(
        mask,
        title=f"Causal Sliding Window (seq={seq_len}, w={window_size})",
        save_path=filename
    )
    print(f"Causal Window {window_size}: {measure_sparsity(mask):.2f}% sparse -> {filename}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Sliding Window Attention Pattern Tests")
    print("=" * 60)
    print()
    
    test_sliding_window_pattern()
    test_causal_sliding_window()
    test_csr_conversion()
    test_attention_correctness()
    test_large_scale_sparsity()
    
    # Optional: visualize (requires matplotlib)
    try:
        visualize_patterns()
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
