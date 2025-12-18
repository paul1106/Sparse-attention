"""
Utility functions for sparse attention patterns.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def attention_dense(Q, K, V, mask=None):
    """
    Dense attention (baseline).
    
    Args:
        Q: [batch, num_heads, seq_len, head_dim]
        K: [batch, num_heads, seq_len, head_dim]
        V: [batch, num_heads, seq_len, head_dim]
        mask: [batch, num_heads, seq_len, seq_len] or None
    
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output


def to_csr(mask):
    """
    Convert boolean mask to CSR format.
    
    Args:
        mask: [seq_len, seq_len] boolean tensor
    
    Returns:
        row_ptr: [seq_len + 1] int tensor
        col_indices: [nnz] int tensor
        nnz: int (number of non-zero elements)
    """
    seq_len = mask.size(0)
    row_ptr = torch.zeros(seq_len + 1, dtype=torch.int32)
    col_indices_list = []
    
    for i in range(seq_len):
        row = mask[i]
        cols = torch.nonzero(row, as_tuple=True)[0]
        col_indices_list.append(cols)
        row_ptr[i + 1] = row_ptr[i] + len(cols)
    
    col_indices = torch.cat(col_indices_list, dim=0).int()
    nnz = len(col_indices)
    
    return row_ptr, col_indices, nnz


def measure_sparsity(mask):
    """
    Measure sparsity percentage.
    
    Args:
        mask: [seq_len, seq_len] boolean tensor
    
    Returns:
        sparsity: float (percentage of zeros)
    """
    total = mask.numel()
    non_zero = mask.sum().item()
    sparsity = (1 - non_zero / total) * 100
    return sparsity


def visualize_sparsity_pattern(mask, title="Sparsity Pattern", save_path=None):
    """
    Visualize sparsity pattern as a matrix plot.
    
    Args:
        mask: [seq_len, seq_len] boolean tensor
        title: str
        save_path: str or None (if provided, save figure)
    """
    mask_np = mask.cpu().numpy()
    seq_len = mask_np.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask_np, cmap='binary', interpolation='nearest')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'{title} ({seq_len}x{seq_len}, {measure_sparsity(mask):.2f}% sparse)')
    
    # Add grid for better visualization
    if seq_len <= 64:
        ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_csr_stats(row_ptr, col_indices):
    """
    Print CSR format statistics.
    
    Args:
        row_ptr: [seq_len + 1] int tensor
        col_indices: [nnz] int tensor
    """
    seq_len = len(row_ptr) - 1
    nnz = len(col_indices)
    total_elements = seq_len * seq_len
    sparsity = (1 - nnz / total_elements) * 100
    
    # Calculate row statistics
    row_nnz = (row_ptr[1:] - row_ptr[:-1]).float()
    max_nnz = row_nnz.max().item()
    min_nnz = row_nnz.min().item()
    avg_nnz = row_nnz.mean().item()
    
    print(f"CSR Format Statistics:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Non-zero Elements: {nnz:,}")
    print(f"  Total Elements: {total_elements:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Row NNZ - Max: {max_nnz:.0f}, Min: {min_nnz:.0f}, Avg: {avg_nnz:.1f}")
    print(f"  Memory (CSR): {(row_ptr.numel() * 4 + col_indices.numel() * 4) / 1024:.2f} KB")
    print(f"  Memory (Dense): {(total_elements * 4) / 1024:.2f} KB")
    print(f"  Memory Reduction: {(1 - (row_ptr.numel() + col_indices.numel()) / total_elements) * 100:.1f}%")
