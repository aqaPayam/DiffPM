import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import sinusoidal_embedding

class StatPredictor(nn.Module):
    """
    Predict per-feature mean and std from positional information.
    
    Args:
        emb_dim (int): Dimensionality of sinusoidal embeddings.
        feature_dim (int): Number of features D in the time series.
    """
    def __init__(self, emb_dim: int, feature_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim

        # Fuse start_idx & series_len embeddings
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Predict 2*D outputs: mu and raw sigma
        self.stat_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * feature_dim)
        )

    def forward(
        self,
        start_idx: torch.Tensor,
        series_len: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            start_idx: (B,) long tensor of window start indices
            series_len: (B,) long tensor of full series lengths

        Returns:
            mu_pred:    (B, D)
            sigma_pred: (B, D) positive via softplus + epsilon
        """
        # 1) Sinusoidal pos-embeddings
        si = sinusoidal_embedding(start_idx.unsqueeze(-1), self.emb_dim)   # (B, emb_dim)
        sl = sinusoidal_embedding(series_len.unsqueeze(-1), self.emb_dim)  # (B, emb_dim)

        # 2) Fuse
        x = torch.cat([si, sl], dim=-1)    # (B, 2*emb_dim)
        x = self.cond_mlp(x)               # (B, emb_dim)

        # 3) Predict stats
        x = self.stat_mlp(x)               # (B, 2*feature_dim)
        mu_pred, sigma_raw = x.chunk(2, dim=-1)  # each (B, feature_dim)

        # 4) Enforce positivity for sigma
        sigma_pred = F.softplus(sigma_raw) + 1e-6

        return mu_pred, sigma_pred
