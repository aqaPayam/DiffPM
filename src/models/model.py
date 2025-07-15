# src/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.diffusion_common.adaptive_norm import AdaptiveNormalizerMLP
from src.models.diffusion_common.embeddings import sinusoidal_embedding, MLPEmbedding


# ------------------------------------------------------------------------------
#  TimeEmbedding: wraps sinusoidal_embedding + small MLP  
# ------------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) long tensor of timesteps
        Returns:
            (B, emb_dim) time embedding
        """
        t = t.unsqueeze(-1)  # (B, 1)
        t_sin = sinusoidal_embedding(t, self.emb_dim)  # (B, emb_dim)
        return self.mlp(t_sin)  # (B, emb_dim)


# ------------------------------------------------------------------------------
#  ConditionalResBlock1D: a 1D residual block with full FiLM (γ, β) conditioning
# ------------------------------------------------------------------------------
class ConditionalResBlock1D(nn.Module):
    """
    A single 1D ResBlock that takes in features (B, C, W) and a condition vector (B, emb_dim).
    It outputs (B, C, W) and applies FiLM (γ⋅h + β) after GroupNorm.
    """
    def __init__(self, in_channels: int, emb_dim: int):
        super().__init__()
        self.C = in_channels
        self.emb_dim = emb_dim

        self.conv1 = nn.Conv1d(self.C, self.C, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=self.C)

        self.conv2 = nn.Conv1d(self.C, self.C, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=self.C)

        # FiLM projection: from emb_dim → 2*C (γ and β)
        self.film_proj = nn.Linear(self.emb_dim, 2 * self.C)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (B, C, W)  feature map
            cond_emb: (B, emb_dim) conditioning embedding
        Returns:
            out:      (B, C, W)
        """
        # Conv1 + GroupNorm
        h = self.conv1(x)   # (B, C, W)
        h = self.gn1(h)     # (B, C, W)

        # FiLM parameters: (B, 2*C)
        film_params = self.film_proj(cond_emb)  # (B, 2C)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, C)

        # Broadcast to (B, C, 1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)    # (B, C, 1)
        h = gamma * h + beta         # (B, C, W)

        # Activation
        h = F.silu(h)

        # Conv2 + GroupNorm
        h2 = self.conv2(h)   # (B, C, W)
        h2 = self.gn2(h2)    # (B, C, W)

        # Residual connection
        return x + h2        # (B, C, W)


# ------------------------------------------------------------------------------
#  ImprovedDiffusionUNet: a true 1D U-Net with FiLM conditioning at each block
# ------------------------------------------------------------------------------
class ImprovedDiffusionUNet(nn.Module):
    """
    A 1D U-Net that predicts noise ε on a normalized window x_t_norm (B, W, D).
    - Three levels of downsampling/upsampling (W → W/2 → W/4 → W/2 → W)
    - FiLM conditioning at every ResBlock using a fused (t, position, stats) embedding.
    """

    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        base_channels: int = 64,
        emb_dim: int = 128,
        n_res_blocks_per_level: int = 2
    ):
        super().__init__()
        self.W = window_size
        self.D = feature_dim
        self.C = base_channels
        self.emb_dim = emb_dim
        self.n_res = n_res_blocks_per_level

        assert self.W % 8 == 0, "window_size must be divisible by 8 for 3 downsamples"

        # Embedding modules (shared for both stat predictor and U-Net conditioning)
        self.time_emb = TimeEmbedding(self.emb_dim)

        # Positional embedding for (start_idx, series_len)
        self.pos_sin_to_emb = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        # Stats embedding for (mu, sigma)
        self.stats_emb = MLPEmbedding(in_dim=2 * self.D, embed_dim=self.emb_dim)

        # Fuse: [t_emb, pos_emb, stats_emb] → emb_dim
        self.fuse_emb = nn.Sequential(
            nn.Linear(3 * self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

        # U-Net layers:

        # Level 1 (full resolution): D → C
        self.in_conv = nn.Conv1d(self.D, self.C, kernel_size=3, padding=1)

        # Down-sample 1: (C → 2C, W → W/2)
        self.down1_conv = nn.Conv1d(self.C, self.C * 2, kernel_size=4, stride=2, padding=1)
        self.down1_resblocks = nn.ModuleList([
            ConditionalResBlock1D(self.C * 2, self.emb_dim)
            for _ in range(self.n_res)
        ])

        # Down-sample 2: (2C → 4C, W/2 → W/4)
        self.down2_conv = nn.Conv1d(self.C * 2, self.C * 4, kernel_size=4, stride=2, padding=1)
        self.down2_resblocks = nn.ModuleList([
            ConditionalResBlock1D(self.C * 4, self.emb_dim)
            for _ in range(self.n_res)
        ])

        # Bottleneck: (4C, W/4)
        self.bot_resblocks = nn.ModuleList([
            ConditionalResBlock1D(self.C * 4, self.emb_dim)
            for _ in range(self.n_res)
        ])

        # Up-sample 1: (4C → 2C, W/4 → W/2)
        self.up1_convtrans = nn.ConvTranspose1d(self.C * 4, self.C * 2, kernel_size=4, stride=2, padding=1)
        self.up1_resblocks = nn.ModuleList([
            ConditionalResBlock1D(self.C * 2, self.emb_dim)
            for _ in range(self.n_res)
        ])

        # Up-sample 2: (2C → C, W/2 → W)
        self.up2_convtrans = nn.ConvTranspose1d(self.C * 2, self.C, kernel_size=4, stride=2, padding=1)
        self.up2_resblocks = nn.ModuleList([
            ConditionalResBlock1D(self.C, self.emb_dim)
            for _ in range(self.n_res)
        ])

        # Final projection: C → D
        self.out_conv = nn.Conv1d(self.C, self.D, kernel_size=3, padding=1)

    def forward(
        self,
        x_t_norm: torch.Tensor,
        start_idx: torch.Tensor,
        series_len: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t_norm  : (B, W, D)    — normalized noisy residual window at time t
            start_idx : (B,)         — start index in original series
            series_len: (B,)         — total length of the full series
            mu        : (B, D)       — windowwise mean (for conditioning)
            sigma     : (B, D)       — windowwise std  (for conditioning)
            t         : (B,)         — timestep index for diffusion

        Returns:
            eps_pred  : (B, W, D)    — predicted noise on normalized scale
        """
        B, W, D = x_t_norm.shape
        assert W == self.W and D == self.D

        # 1) Build the fused conditioning embedding (B, emb_dim)
        # 1a) Time embedding
        t_emb = self.time_emb(t)  # (B, emb_dim)

        # 1b) Positional embedding for (start_idx, series_len)
        si_sin = sinusoidal_embedding(start_idx.unsqueeze(-1), self.emb_dim)  # (B, emb_dim)
        sl_sin = sinusoidal_embedding(series_len.unsqueeze(-1), self.emb_dim) # (B, emb_dim)
        pos_cat = torch.cat([si_sin, sl_sin], dim=-1)  # (B, 2*emb_dim)
        pos_emb = self.pos_sin_to_emb(pos_cat)         # (B, emb_dim)

        # 1c) Stats embedding from (mu, sigma)
        stats_cat = torch.cat([mu, sigma], dim=-1)      # (B, 2*D)
        stats_emb = self.stats_emb(stats_cat)           # (B, emb_dim)

        # 1d) Fuse them: [t_emb, pos_emb, stats_emb] → (B, 3*emb_dim) → (B, emb_dim)
        fuse_input = torch.cat([t_emb, pos_emb, stats_emb], dim=-1)  # (B, 3*emb_dim)
        cond_emb = self.fuse_emb(fuse_input)                           # (B, emb_dim)

        # 2) U-Net forward

        # Permute to (B, D, W) for 1D conv
        h = x_t_norm.permute(0, 2, 1)  # (B, D, W)

        # 2a) Input conv: D → C
        h0 = self.in_conv(h)          # (B, C, W)

        # 2b) Down-sample #1: (C → 2C, W → W/2)
        d1 = self.down1_conv(F.silu(h0))  # (B, 2C, W/2)
        for block in self.down1_resblocks:
            d1 = block(d1, cond_emb)      # (B, 2C, W/2)

        # 2c) Down-sample #2: (2C → 4C, W/2 → W/4)
        d2 = self.down2_conv(F.silu(d1))  # (B, 4C, W/4)
        for block in self.down2_resblocks:
            d2 = block(d2, cond_emb)       # (B, 4C, W/4)

        # 2d) Bottleneck (4C, W/4)
        b = d2
        for block in self.bot_resblocks:
            b = block(b, cond_emb)         # (B, 4C, W/4)

        # 2e) Up-sample #1: (4C → 2C, W/4 → W/2)
        u1 = self.up1_convtrans(F.silu(b))   # (B, 2C, W/2)
        u1 = u1 + d1                         # skip-connection
        for block in self.up1_resblocks:
            u1 = block(u1, cond_emb)         # (B, 2C, W/2)

        # 2f) Up-sample #2: (2C → C, W/2 → W)
        u2 = self.up2_convtrans(F.silu(u1))  # (B, C, W)
        u2 = u2 + h0                         # skip-connection
        for block in self.up2_resblocks:
            u2 = block(u2, cond_emb)         # (B, C, W)

        # 2g) Final projection: C → D
        out = self.out_conv(F.silu(u2))      # (B, D, W)

        # Permute back to (B, W, D)
        eps_pred = out.permute(0, 2, 1)       # (B, W, D)
        return eps_pred


# ------------------------------------------------------------------------------
#  StatPredictor: predicts (mu, sigma) from fused condition embedding
# ------------------------------------------------------------------------------
class StatPredictor(nn.Module):
    """
    Given the fused condition embedding of (t, start_idx, series_len),
    this MLP outputs a predicted (mu, sigma) for each feature.
    """
    def __init__(self, emb_dim: int, feature_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.SiLU(),
            nn.Linear(self.emb_dim, 2 * self.feature_dim)
        )

    def forward(self, cond_emb: torch.Tensor):
        """
        Args:
            cond_emb: (B, emb_dim)
        Returns:
            mu_pred    : (B, D)
            sigma_pred : (B, D)  (strictly positive via softplus + epsilon)
        """
        x = self.mlp(cond_emb)       # (B, 2*D)
        mu_pred, sigma_pred = x.chunk(2, dim=-1)  # each (B, D)
        sigma_pred = F.softplus(sigma_pred) + 1e-6
        return mu_pred, sigma_pred     # both (B, D)


# ------------------------------------------------------------------------------
#  DiffusionModel: integrates AdaptiveNormalizer, StatPredictor, and U-Net
# ------------------------------------------------------------------------------
class DiffusionModel(nn.Module):
    """
    Top-level model that:
      - Learns to predict (mu, sigma) from (t, start_idx, series_len)
      - Uses those (mu, sigma) to normalize raw windows via AdaptiveNormalizerMLP
      - Adds diffusion noise in normalized space
      - Uses a U-Net (ImprovedResidualDiffusionUNet) with FiLM conditioning (using predicted stats)
        to predict noise
      - Returns denoising loss + stat-prediction loss
    """
    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        base_channels: int = 64,
        emb_dim: int = 128,
        n_res_blocks_per_level: int = 2
    ):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Shared embedding modules inside the U-Net (we will reuse them for conditioning & stat prediction)
        self.unet = ImprovedDiffusionUNet(
            window_size=window_size,
            feature_dim=feature_dim,
            base_channels=base_channels,
            emb_dim=emb_dim,
            n_res_blocks_per_level=n_res_blocks_per_level
        )

        # Stat predictor head: (B, emb_dim) → (B, 2*D)
        self.stat_predictor = StatPredictor(emb_dim=emb_dim, feature_dim=feature_dim)

        # Adaptive normalizer: (x_raw, mu, sigma) → (x_norm, {'alpha', 'beta'})
        self.adaptive_norm = AdaptiveNormalizerMLP(feature_dim, hidden_dim=feature_dim * 2)

    def forward(
        self,
        x_raw: torch.Tensor,
        start_idx: torch.Tensor,
        series_len: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
        alpha_bar_t: torch.Tensor = None  # shape (B,)
    ):
        """
        Args:
            x_raw      : (B, W, D)  -- raw windows (training only)
            start_idx  : (B,)       -- start index of each window
            series_len : (B,)       -- total series length
            t          : (B,)       -- timestep indices
            noise      : (B, W, D)  -- if not provided, sample N(0,1)
            alpha_bar_t: (B,)       -- sqrt(alpha_bar_t) if provided, else must be computed externally

        Returns:
            total_loss     : scalar
            denoise_loss   : scalar
            stat_loss      : scalar
            mu_true        : (B, D)
            sigma_true     : (B, D)
            mu_pred        : (B, D)
            sigma_pred     : (B, D)
        """
        B, W, D = x_raw.shape
        assert W == self.window_size and D == self.feature_dim

        # 1) Compute true window stats
        mu_true = x_raw.mean(dim=1)                                             # (B, D)
        sigma_true = x_raw.std(dim=1, unbiased=False) + 1e-6                     # (B, D)

        # 2) Build fused condition embedding: (B, emb_dim)
        t_emb = self.unet.time_emb(t)  # (B, emb_dim)

        si_sin = sinusoidal_embedding(start_idx.unsqueeze(-1), self.unet.emb_dim)
        sl_sin = sinusoidal_embedding(series_len.unsqueeze(-1), self.unet.emb_dim)
        pos_cat = torch.cat([si_sin, sl_sin], dim=-1)  # (B, 2*emb_dim)
        pos_emb = self.unet.pos_sin_to_emb(pos_cat)     # (B, emb_dim)

        stats_cat_true = torch.cat([mu_true, sigma_true], dim=-1)  # (B, 2*D)
        stats_emb_true = self.unet.stats_emb(stats_cat_true)      # (B, emb_dim)

        fuse_input = torch.cat([t_emb, pos_emb, stats_emb_true], dim=-1)  # (B, 3*emb_dim)
        cond_emb = self.unet.fuse_emb(fuse_input)                          # (B, emb_dim)

        # 3) Stat predictor: predict (mu_pred, sigma_pred) from cond_emb
        mu_pred, sigma_pred = self.stat_predictor(cond_emb)  # each (B, D)

        # 4) Use true (mu_true, sigma_true) to normalize raw window
        x_norm, norm_params = self.adaptive_norm(x_raw, mu_true, sigma_true)
        # norm_params = {'alpha': (B, D), 'beta': (B, D)}

        # 5) Sample noise in normalized space if not given
        if noise is None:
            noise = torch.randn_like(x_norm)  # (B, W, D)

        # 6) Form x_t_norm = sqrt(alpha_bar_t) * x_norm + sqrt(1 - alpha_bar_t) * noise
        #    alpha_bar_t must be provided from outside (computed via cosine_beta_schedule).
        assert alpha_bar_t is not None, "You must supply alpha_bar_t (shape (B,))"
        sqrt_ab = torch.sqrt(alpha_bar_t).view(B, 1, 1)           # (B,1,1)
        sqrt_omb = torch.sqrt(1.0 - alpha_bar_t).view(B, 1, 1)    # (B,1,1)
        x_t_norm = sqrt_ab * x_norm + sqrt_omb * noise           # (B, W, D)

        # 7) U-Net denoising: but now condition on predicted (mu_pred, sigma_pred)
        eps_pred = self.unet(
            x_t_norm,
            start_idx,
            series_len,
            mu_pred,      # use predicted stats
            sigma_pred,   # use predicted stats
            t
        )  # (B, W, D)

        # 8) Loss terms
        denoise_loss = F.mse_loss(eps_pred, noise)
        stat_loss = F.mse_loss(mu_pred, mu_true) + F.mse_loss(sigma_pred, sigma_true)
        # Weighting factor λ for stat prediction loss
        lambda_stat = 0.1
        total_loss = denoise_loss + lambda_stat * stat_loss

        return total_loss, denoise_loss, stat_loss, mu_true, sigma_true, mu_pred, sigma_pred