#src/models/diffusion_common/embeddings.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings.

    Args:
        x (Tensor): LongTensor or FloatTensor of shape (B,) or (B,1) with position indices.
        dim (int): Embedding dimension.

    Returns:
        Tensor of shape (B, dim) with sinusoidal embeddings.
    """
    # Flatten to (B,)
    if x.dim() == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    device = x.device
    half_dim = dim // 2
    # Create frequency vector
    freq_seq = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=device) / half_dim)
    args = x.float().unsqueeze(1) * freq_seq.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half_dim)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb  # shape (B, dim)

class MLPEmbedding(nn.Module):
    """
    Simple MLP-based embedding for conditioning vectors.

    Args:
        in_dim (int): Dimension of input features.
        embed_dim (int): Desired embedding output dimension.
    """
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, in_dim).
        Returns:
            Tensor of shape (B, embed_dim).
        """
        return self.net(x)

