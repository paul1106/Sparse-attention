"""
Generate test data for sparse attention benchmarks.

Creates Q, K, V tensors at various scales:
- Small: 1K tokens (for testing)
- Medium: 8K tokens (Llama-2 context)
- Large: 32K tokens (Llama-3 extended)
- Extra Large: 128K tokens (Llama-3 405B)
"""

import torch
import os


def generate_qkv(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda'):
    """
    Generate random Q, K, V tensors.
    
    Args:
        batch_size: int
        num_heads: int
        seq_len: int
        head_dim: int
        dtype: torch dtype
        device: str
    
    Returns:
        Q, K, V: [batch_size, num_heads, seq_len, head_dim]
    """
    print(f"Generating Q, K, V with:")
    print(f"  Batch: {batch_size}, Heads: {num_heads}, Seq: {seq_len}, Dim: {head_dim}")
    print(f"  Dtype: {dtype}, Device: {device}")
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Normalize (similar to real embeddings)
    Q = Q / (head_dim ** 0.5)
    K = K / (head_dim ** 0.5)
    
    total_memory = (Q.numel() + K.numel() + V.numel()) * 2  # fp16 = 2 bytes
    print(f"  Total memory: {total_memory / (1024**3):.2f} GB")
    
    return Q, K, V


def save_test_data(Q, K, V, filename):
    """Save Q, K, V to file."""
    data = {
        'Q': Q.cpu(),
        'K': K.cpu(),
        'V': V.cpu(),
    }
    torch.save(data, filename)
    print(f"Saved to {filename}")
    
    # Print file size
    file_size = os.path.getsize(filename)
    print(f"  File size: {file_size / (1024**2):.2f} MB\n")


def main():
    """Generate test data at various scales."""
    
    # Standard Llama-3 config
    batch_size = 1
    num_heads = 32  # Llama-3 8B: 32 heads
    head_dim = 128  # Standard head dimension
    
    configs = [
        ("1k", 1024),
        ("2k", 2048),
        ("4k", 4096),
        ("8k", 8192),
        ("16k", 16384),
        ("32k", 32768),
        ("64k", 65536),
        ("128k", 131072),
    ]
    
    print("=" * 60)
    print("Generating Test Data for Sparse Attention Benchmarks")
    print("=" * 60)
    print()
    
    for name, seq_len in configs:
        print(f"Generating {name} test data...")
        try:
            Q, K, V = generate_qkv(batch_size, num_heads, seq_len, head_dim)
            save_test_data(Q, K, V, f"test_data_{name}.pt")
        except RuntimeError as e:
            print(f"  ❌ Failed: {e}")
            print(f"  (Out of memory for {name}, skipping)\n")
    
    print("=" * 60)
    print("Done! Test data generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
