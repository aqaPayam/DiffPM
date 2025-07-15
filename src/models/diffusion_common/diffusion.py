#src/models/diffusion_common/diffusion.py

import torch
import torch.nn.functional as F
import math

def cosine_beta_schedule(T: int, s: float = 0.008):
    """
    Cosine schedule for betas, from Nichol & Dhariwal (2021).

    Args:
        T (int): Number of diffusion timesteps.
        s (float): Small offset to prevent singularities.

    Returns:
        betas (Tensor[T]), alphas (Tensor[T]), alpha_bars (Tensor[T]).
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.clamp(betas, max=0.999)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

class GaussianDiffusion:
    """
    Core diffusion class implementing forward & reverse processes.

    Args:
        betas (Tensor): Noise schedule of shape (T,).
        device (torch.device): Device to run operations on.
    """
    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.device = device
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.num_timesteps = betas.shape[0]

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Adds noise to x_start at step t (forward diffusion).

        x_t = sqrt(alpha_bar_t)*x_start + sqrt(1-alpha_bar_t)*noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # reshape t for broadcasting: (B,) → (B, 1, 1)
        shape = [t.size(0)] + [1] * (x_start.dim() - 1)
        sqrt_ab = self.sqrt_alpha_bars[t].view(shape)
        sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(shape)
        return sqrt_ab * x_start + sqrt_omb * noise

    def p_losses(self, model, x_start: torch.Tensor, cond: dict, t: torch.Tensor):
        """
        Compute the training loss (MSE on noise prediction).

        Args:
            model: The diffusion U-Net model.
            x_start: Original data tensor (B, ..., D).
            cond: Conditioning dict with keys 'mean','std','start_idx','series_len'.
            t: Tensor of shape (B,) with timestep indices.
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        # Predict noise
        eps_pred = model(
            x_noisy,
            cond['mean'].to(self.device),
            cond['std'].to(self.device),
            cond['start_idx'].to(self.device),
            cond['series_len'].to(self.device),
            t
        )
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, cond: dict, t: torch.Tensor):
        """
        Sample one reverse diffusion step from x_t → x_{t-1}.
        """
        eps_pred = model(
            x,
            cond['mean'],
            cond['std'],
            cond['start_idx'],
            cond['series_len'],
            t
        )
        beta_t = self.betas[t].view(-1, *([1] * (x.dim() - 1)))
        alpha_t = self.alphas[t].view(-1, *([1] * (x.dim() - 1)))
        alpha_bar_t = self.alpha_bars[t].view(-1, *([1] * (x.dim() - 1)))

        # posterior mean
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x - coef2 * eps_pred)

        # noise for non-zero t
        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, *([1] * (x.dim() - 1)))
        return mean + mask * torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample(self, model, cond: dict, shape: tuple):
        """
        Generate samples by iteratively applying p_sample from t=T-1 to t=0.

        Args:
            model: Trained diffusion model.
            cond: Conditioning dict, containing per-sample 'mean','std','start_idx','series_len'.
            shape: Desired output shape (B, ..., D).
        Returns:
            Generated tensor of shape `shape`.
        """
        x = torch.randn(shape, device=self.device)
        B = shape[0]
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, cond, t)
        return x
