import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .schedule import cosine_beta_schedule
from .stat_predictor import StatPredictor
from .unet import ImprovedDiffusionUNet1D

class DiffusionModel(nn.Module):
    """
    Top-level diffusion model that:
      1. Predicts per-window mean & std from (start_idx, series_len)
      2. Normalizes raw windows using those stats
      3. Adds noise according to a cosine schedule
      4. Uses the provided single-channel 1D U-Net to predict that noise
      5. Computes denoising + stat-prediction losses
      6. Provides a `sample()` method for inference

    All hyperparameters are passed explicitly to __init__; no defaults.
    NOTE: this assumes feature_dim == 1, since the U-Net takes a (B, W) input.
    """
    def __init__(
        self,
        window_size: int,
        feature_dim: int,
        emb_dim: int,
        base_channels: int,
        num_res_blocks: int,
        num_timesteps: int,
        schedule_s: float,
        stat_loss_weight: float
    ):
        super().__init__()
        assert feature_dim == 1, "ImprovedDiffusionUNet1D supports only single-channel input"

        # 1) Stat predictor
        self.stat_predictor = StatPredictor(
            emb_dim=emb_dim,
            feature_dim=feature_dim
        )

        # 2) Single-channel 1D U-Net
        self.unet = ImprovedDiffusionUNet1D(
            window_size=window_size,
            time_emb_dim=emb_dim,
            base_channels=base_channels,
            n_res_blocks=num_res_blocks
        )

        # 3) Noise schedule
        betas, alphas, alpha_bars = cosine_beta_schedule(
            num_timesteps=num_timesteps,
            s=schedule_s
        )
        self.register_buffer('betas',      betas)
        self.register_buffer('alphas',     alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        # 4) Weight for the stat-prediction loss
        self.stat_loss_weight = stat_loss_weight

    def forward(
        self,
        x_raw: torch.Tensor,
        start_idx: torch.Tensor,
        series_len: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_raw      : (B, W, 1)  raw unnormalized windows
            start_idx  : (B,)       start indices
            series_len : (B,)       full series lengths
            t          : (B,)       diffusion step indices
            noise      : optional (B, W, 1), else sampled internally

        Returns:
            total_loss   : scalar
            denoise_loss : scalar
            stat_loss    : scalar
        """
        B, W, D = x_raw.shape
        assert D == 1
        assert W == self.unet.W

        # -- 1) Stat prediction
        mu_pred, sigma_pred = self.stat_predictor(start_idx, series_len)  # (B,1)

        # -- 2) True stats for loss
        mu_true    = x_raw.mean(dim=1)                              # (B,1)
        sigma_true = x_raw.std(dim=1, unbiased=False) + 1e-6        # (B,1)

        # -- 3) Normalize
        x_norm = (x_raw - mu_pred.unsqueeze(1)) / sigma_pred.unsqueeze(1)  # (B, W, 1)

        # -- 4) Noise sampling
        if noise is None:
            noise = torch.randn_like(x_norm)  # (B, W, 1)

        # -- 5) Forward diffusion (normalized space)
        alpha_bar_t = self.alpha_bars[t]                       # (B,)
        sqrt_ab     = torch.sqrt(alpha_bar_t).view(B, 1, 1)    # (B,1,1)
        sqrt_omb    = torch.sqrt(1.0 - alpha_bar_t).view(B,1,1)
        x_t_norm    = sqrt_ab * x_norm + sqrt_omb * noise      # (B, W, 1)

        # -- 6) Predict noise (single-channel)
        x_in = x_t_norm.squeeze(-1)                            # (B, W)
        eps_pred_1ch = self.unet(
            x=x_in,
            t=t,
            start_idx=start_idx,
            series_len=series_len
        )  # (B, W)
        eps_pred = eps_pred_1ch.unsqueeze(-1)                  # (B, W, 1)

        # -- 7) Losses
        denoise_loss = F.mse_loss(eps_pred, noise)
        stat_loss    = F.mse_loss(mu_pred,   mu_true) \
                     + F.mse_loss(sigma_pred, sigma_true)
        total_loss   = denoise_loss + self.stat_loss_weight * stat_loss

        return total_loss, denoise_loss, stat_loss

    @torch.no_grad()
    def sample(
        self,
        start_idx: torch.Tensor,
        series_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate samples from the trained diffusion model.

        Args:
            start_idx:  (B,) start indices
            series_len: (B,) full series lengths

        Returns:
            x_sample:   (B, W, 1) denoised, de-normalized time-series
        """
        device = start_idx.device
        B = start_idx.size(0)
        W = self.unet.W

        # 1) Predict normalization stats
        mu_pred, sigma_pred = self.stat_predictor(start_idx, series_len)  # (B,1)

        # 2) Initialize pure noise in normalized space
        x = torch.randn(B, W, 1, device=device)

        # 3) Reverse diffusion loop
        T = self.alpha_bars.size(0)
        for i in reversed(range(T)):
            t = torch.full((B,), i, dtype=torch.long, device=device)  # (B,)

            # Predict noise
            x_in = x.squeeze(-1)                                    # (B, W)
            eps_pred_1ch = self.unet(
                x=x_in,
                t=t,
                start_idx=start_idx,
                series_len=series_len
            )                                                        # (B, W)
            eps_pred = eps_pred_1ch.unsqueeze(-1)                   # (B, W, 1)

            # Gather scalars
            beta_t      = self.betas[t].view(B,1,1)                 # (B,1,1)
            alpha_t     = self.alphas[t].view(B,1,1)                # (B,1,1)
            alpha_bar_t = self.alpha_bars[t].view(B,1,1)            # (B,1,1)

            # Compute posterior mean
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps_pred)                   # (B, W, 1)

            # Add noise if not final step
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise               # (B, W, 1)
            else:
                x = mean

        # 4) De-normalize
        x_sample = x * sigma_pred.unsqueeze(1) + mu_pred.unsqueeze(1)  # (B, W, 1)
        return x_sample
