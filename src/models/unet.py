import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import sinusoidal_embedding as get_sinusoidal_embedding

class ResBlock1D(nn.Module):
    def __init__(self, c: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.gn1   = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        self.gn2   = nn.GroupNorm(8, c)
        self.film  = nn.Linear(emb_dim, c)  # produce an additive bias

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, C, W)
        emb: (B, emb_dim)
        """
        h = self.conv1(x)
        h = self.gn1(h)
        bias = self.film(emb).unsqueeze(-1)  # (B, C, 1)
        h = h + bias
        h = F.silu(h)

        h = self.conv2(h)
        h = self.gn2(h)
        return x + h  # residual


class ImprovedDiffusionUNet1D(nn.Module):
    def __init__(
        self,
        window_size: int,
        time_emb_dim: int,
        base_channels: int,
        n_res_blocks: int
    ):
        super().__init__()
        self.W = window_size
        self.base_c = base_channels
        self.time_emb_dim = time_emb_dim

        # input conv: single-channel in → C channels
        self.init_conv = nn.Conv1d(1, base_channels, 3, padding=1)

        # MLP to process sinusoidal time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # MLP to fuse start_idx & series_len embeddings
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock1D(base_channels, time_emb_dim)
            for _ in range(n_res_blocks)
        ])

        # output conv: C channels → single channel out
        self.out_conv = nn.Conv1d(base_channels, 1, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        start_idx: torch.Tensor,
        series_len: torch.Tensor
    ) -> torch.Tensor:
        """
        x:          (B, W)         — normalized, noised window (single-channel)
        t:          (B,)           — timestep indices
        start_idx:  (B,)           — window start indices
        series_len: (B,)           — full series lengths

        returns:
        eps_pred:   (B, W)         — predicted noise (single-channel)
        """
        B, W = x.shape
        assert W == self.W, f"expected width {self.W}, got {W}"

        # 1) Input projection
        h = x.unsqueeze(1)                     # (B, 1, W)
        h = self.init_conv(h)                  # (B, C, W)

        # 2) Time embedding
        t_emb = get_sinusoidal_embedding(t.unsqueeze(-1), self.time_emb_dim)  # (B, E)
        t_emb = self.time_mlp(t_emb)                                          # (B, E)

        # 3) Position embedding
        se = get_sinusoidal_embedding(start_idx.unsqueeze(-1), self.time_emb_dim)
        le = get_sinusoidal_embedding(series_len.unsqueeze(-1), self.time_emb_dim)
        cond = torch.cat([se, le], dim=-1)    # (B, 2E)
        cond_emb = self.cond_mlp(cond)         # (B, E)

        # 4) Combine time + pos
        emb = t_emb + cond_emb                 # (B, E)

        # 5) Residual blocks
        for block in self.res_blocks:
            h = block(h, emb)                  # (B, C, W)

        # 6) Output projection
        out = self.out_conv(F.silu(h))        # (B, 1, W)
        eps_pred = out.squeeze(1)             # (B, W)

        return eps_pred
