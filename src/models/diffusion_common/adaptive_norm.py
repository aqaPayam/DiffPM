# src/models/diffusion_common/adaptive_norm.py

import torch
import torch.nn as nn

class AdaptiveNormalizerMLP(nn.Module):
    """
    Adaptive Normalizer module for dynamic window-wise normalization.

    This module learns to predict scaling (alpha) and shifting (beta)
    parameters for each feature dimension based on the window's mean and std.

    Forward normalization:
        x_norm = (x - mu) / sigma * alpha + beta

    Inverse normalization:
        x = ((x_norm - beta) / alpha) * sigma + mu

    Args:
        feature_dim (int): Number of features (channels) per time-step.
        hidden_dim (int): Hidden dimension for the MLP.
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        # MLP takes concatenated [mu, sigma] of size 2*feature_dim, outputs alpha and beta of size 2*feature_dim
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)
        )

    def forward(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        """
        Forward normalization.

        Args:
            x (Tensor): Input tensor of shape (B, T, D) or (B, D, T).
            mu (Tensor): Window-wise mean, shape (B, D).
            sigma (Tensor): Window-wise std, shape (B, D).

        Returns:
            x_norm (Tensor): Normalized tensor, same shape as x.
            params (dict): {'alpha': Tensor(B, D), 'beta': Tensor(B, D)} for inversion.
        """
        # Pack stats
        stats = torch.cat([mu, sigma], dim=-1)    # (B, 2D)
        alpha_beta = self.net(stats)              # (B, 2D)
        alpha, beta = alpha_beta.chunk(2, dim=-1) # each (B, D)

        # Broadcast for normalization
        if x.dim() == 3 and x.size(-1) == self.feature_dim:
            # x: (B, T, D)
            mu_b, sigma_b = mu.unsqueeze(1), sigma.unsqueeze(1)
            alpha_b, beta_b = alpha.unsqueeze(1), beta.unsqueeze(1)
            x_norm = (x - mu_b) / sigma_b * alpha_b + beta_b

        elif x.dim() == 3 and x.size(1) == self.feature_dim:
            # x: (B, D, T)
            mu_b, sigma_b = mu.unsqueeze(-1), sigma.unsqueeze(-1)
            alpha_b, beta_b = alpha.unsqueeze(-1), beta.unsqueeze(-1)
            x_norm = (x - mu_b) / sigma_b * alpha_b + beta_b

        else:
            raise ValueError(
                f"AdaptiveNormalizerMLP: expected x of shape (B, T, {self.feature_dim}) "
                f"or (B, {self.feature_dim}, T), got {tuple(x.shape)}"
            )

        return x_norm, {'alpha': alpha, 'beta': beta}

    def inverse(self, x_norm: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, params: dict):
        """
        Inverse normalization to reconstruct original scale.

        Args:
            x_norm (Tensor): Normalized tensor, same shape as original x.
            mu (Tensor): Window-wise mean, shape (B, D).
            sigma (Tensor): Window-wise std, shape (B, D).
            params (dict): {'alpha': Tensor(B, D), 'beta': Tensor(B, D)}

        Returns:
            x_recon (Tensor): Reconstructed tensor in original scale.
        """
        alpha, beta = params['alpha'], params['beta']

        if x_norm.dim() == 3 and x_norm.size(-1) == self.feature_dim:
            mu_b, sigma_b = mu.unsqueeze(1), sigma.unsqueeze(1)
            alpha_b, beta_b = alpha.unsqueeze(1), beta.unsqueeze(1)
            x_recon = ((x_norm - beta_b) / alpha_b) * sigma_b + mu_b

        elif x_norm.dim() == 3 and x_norm.size(1) == self.feature_dim:
            mu_b, sigma_b = mu.unsqueeze(-1), sigma.unsqueeze(-1)
            alpha_b, beta_b = alpha.unsqueeze(-1), beta.unsqueeze(-1)
            x_recon = ((x_norm - beta_b) / alpha_b) * sigma_b + mu_b

        else:
            raise ValueError(
                f"AdaptiveNormalizerMLP.inverse: expected x_norm of shape "
                f"(B, T, {self.feature_dim}) or (B, {self.feature_dim}, T), got {tuple(x_norm.shape)}"
            )

        return x_recon
