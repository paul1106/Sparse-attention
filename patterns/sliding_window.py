"""
Sliding Window Attention Pattern.

Used in: Llama-3, Mistral, Longformer
Pattern: Each token attends to ±W neighbors
Sparsity: ~99% for W=4096, n=128K
"""

import torch
import torch.nn.functional as F


def sliding_window_mask(seq_len, window_size, device='cuda'):
    """
    Create sliding window attention mask.
    
    Args:
        seq_len: int, sequence length
        window_size: int, window size (attends to ±window_size)
        device: str, 'cuda' or 'cpu'
    
    Returns:
        mask: [seq_len, seq_len] boolean tensor
              True where attention is allowed, False otherwise
    
    Example:
        seq_len = 8, window_size = 2
        mask[i, j] = True if |i - j| <= 2
        
        Pattern (1 = attend, 0 = ignore):
        [[1 1 1 0 0 0 0 0]
         [1 1 1 1 0 0 0 0]
         [1 1 1 1 1 0 0 0]
         [0 1 1 1 1 1 0 0]
         [0 0 1 1 1 1 1 0]
         [0 0 0 1 1 1 1 1]
         [0 0 0 0 1 1 1 1]
         [0 0 0 0 0 1 1 1]]
    """
    # Create position indices
    positions = torch.arange(seq_len, device=device)
    
    # Calculate distance matrix
    distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
    
    # Mask where distance <= window_size
    mask = distances <= window_size
    
    return mask


def sliding_window_attention(Q, K, V, window_size):
    """
    Sliding window attention (PyTorch implementation).
    
    Args:
        Q: [batch, num_heads, seq_len, head_dim]
        K: [batch, num_heads, seq_len, head_dim]
        V: [batch, num_heads, seq_len, head_dim]
        window_size: int
    
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    batch, num_heads, seq_len, head_dim = Q.shape
    device = Q.device
    
    # Create sliding window mask
    mask = sliding_window_mask(seq_len, window_size, device=device)
    
    # Compute attention scores
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask
    scores = scores.masked_fill(~mask, float('-inf'))
    
    # Softmax and attention
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output


def sliding_window_causal_mask(seq_len, window_size, device='cuda'):
    """
    Create causal sliding window attention mask (for autoregressive models).
    
    Args:
        seq_len: int
        window_size: int
        device: str
    
    Returns:
        mask: [seq_len, seq_len] boolean tensor
              True where attention is allowed (both within window AND causal)
    
    Example:
        seq_len = 8, window_size = 2
        
        Pattern (1 = attend, 0 = ignore):
        [[1 0 0 0 0 0 0 0]    # Can only see self
         [1 1 0 0 0 0 0 0]    # Can see [0, 1]
         [1 1 1 0 0 0 0 0]    # Can see [0, 1, 2]
         [0 1 1 1 0 0 0 0]    # Can see [1, 2, 3] (window constraint)
         [0 0 1 1 1 0 0 0]    # Can see [2, 3, 4]
         [0 0 0 1 1 1 0 0]    # Can see [3, 4, 5]
         [0 0 0 0 1 1 1 0]    # Can see [4, 5, 6]
         [0 0 0 0 0 1 1 1]]   # Can see [5, 6, 7]
    """
    # Start with sliding window mask
    mask = sliding_window_mask(seq_len, window_size, device=device)
    
    # Apply causal constraint (can only attend to past and present)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    
    # Combine both constraints
    mask = mask & causal_mask
    
    return mask


def sliding_window_attention_causal(Q, K, V, window_size):
    """
    Causal sliding window attention (for autoregressive generation).
    
    Args:
        Q: [batch, num_heads, seq_len, head_dim]
        K: [batch, num_heads, seq_len, head_dim]
        V: [batch, num_heads, seq_len, head_dim]
        window_size: int
    
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    batch, num_heads, seq_len, head_dim = Q.shape
    device = Q.device
    
    # Create causal sliding window mask
    mask = sliding_window_causal_mask(seq_len, window_size, device=device)
    
    # Compute attention scores
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask
    scores = scores.masked_fill(~mask, float('-inf'))
    
    # Softmax and attention
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output
