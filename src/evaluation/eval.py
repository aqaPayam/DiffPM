# inference/eval.py

import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_handling.data import load_dataset
from src.data_handling.decomposition import split_trend_residual
from src.models.diffusion_common.diffusion import cosine_beta_schedule
from src.models.model import ImprovedDiffusionUNet
from src.models.diffusion_common.adaptive_norm import AdaptiveNormalizerMLP


def reconstruct_window(
    unet: torch.nn.Module,
    window_np: np.ndarray,
    alpha_bars: torch.Tensor,
    t: int,
    device: torch.device
) -> np.ndarray:
    """
    Reconstruct a single window (shape (W, D)) from a trained UNet at timestep t.
    Returns a NumPy array of shape (W, D).
    """
    W, D = window_np.shape

    # 1) Compute true window stats
    mu_np = window_np.mean(axis=0)                  # (D,)
    sigma_np = window_np.std(axis=0, ddof=0) + 1e-6  # (D,)

    # 2) Build torch tensors for x_raw, mu, sigma, start_idx, series_len
    x_raw = torch.from_numpy(window_np).unsqueeze(0).float().to(device)  # (1, W, D)
    mu    = torch.from_numpy(mu_np).unsqueeze(0).float().to(device)      # (1, D)
    sigma = torch.from_numpy(sigma_np).unsqueeze(0).float().to(device)   # (1, D)
    start_idx = torch.tensor([0], dtype=torch.long, device=device)      # (1,)
    series_len = torch.tensor([W], dtype=torch.long, device=device)     # (1,)

    # 3) Normalize x_raw via AdaptiveNormalizerMLP
    norm = AdaptiveNormalizerMLP(feature_dim=D, hidden_dim=D * 2).to(device)
    with torch.no_grad():
        x_norm, _ = norm(x_raw, mu, sigma)  # (1, W, D)

    # 4) Sample a fixed noise vector
    torch.manual_seed(0)
    noise = torch.randn_like(x_norm)  # (1, W, D)

    # 5) Build x_t_norm
    alpha_bar_t = alpha_bars[t].view(1, 1, 1)  # (1,1,1)
    sqrt_ab = torch.sqrt(alpha_bar_t)
    sqrt_omb = torch.sqrt(1.0 - alpha_bar_t)
    x_t_norm = sqrt_ab * x_norm + sqrt_omb * noise  # (1, W, D)

    # 6) Predict eps via UNet, then reconstruct x0_norm → x0_pred
    with torch.no_grad():
        eps_pred = unet(x_t_norm, start_idx, series_len, mu, sigma, torch.tensor([t], device=device))
        x0_norm_pred = (x_t_norm - sqrt_omb * eps_pred) / sqrt_ab  # (1, W, D)
        x0_pred = x0_norm_pred * sigma.unsqueeze(1) + mu.unsqueeze(1)  # (1, W, D)

    return x0_pred.squeeze(0).cpu().numpy()  # shape (W, D)


def evaluate_reconstruction(
    dataset_name: str,
    feature_idx: int,
    trend_config: str,
    resid_config: str,
    trend_ckpt: str,
    resid_ckpt: str,
    out_dir: str,
    device: torch.device = None
):
    """
    Performs reconstruction on a single meaningful chunk from both the Residual UNet and Trend UNet.
    - Loads one real series (batch_size=1) from `dataset_name`.
    - Decomposes into trend & residual (moving‐average).
    - Selects the middle window of length `window_size` from each component.
    - Loads the trained UNet weights (from `trend_ckpt` and `resid_ckpt`).
    - Builds diffusion schedules from each YAML’s `diffusion.steps`.
    - Uses t = T//2 for reconstruction.
    - Reconstructs that window at t=T//2, computes MSE, and plots “original vs. reconstructed.”
    - Saves plots and writes a small “mse_report.txt” in `out_dir`.
    """

    os.makedirs(out_dir, exist_ok=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # 1) Load and decompose real series
    data = load_dataset(dataset_name, batch_size=1)       # (1, T_real, D)
    real = data[0]                                        # (T_real, D)
    T_real, D = real.shape

    trend_list, resid_list = split_trend_residual(real[np.newaxis, ...])
    real_trend = trend_list[0]    # (T_real, D)
    real_resid  = resid_list[0]   # (T_real, D)

    # 2) Load window_size from each config and select the middle window
    resid_cfg = yaml.safe_load(open(resid_config))
    W_r = resid_cfg['residual']['window_size']
    assert T_real >= W_r, f"T_real ({T_real}) must be >= resid window_size ({W_r})"
    mid_r = (T_real - W_r) // 2
    win_resid = real_resid[mid_r : mid_r + W_r, :]  # (W_r, D)

    trend_cfg = yaml.safe_load(open(trend_config))
    W_t = trend_cfg['trend']['window_size']
    assert T_real >= W_t, f"T_real ({T_real}) must be >= trend window_size ({W_t})"
    mid_t = (T_real - W_t) // 2
    win_trend = real_trend[mid_t : mid_t + W_t, :]  # (W_t, D)

    # 3) Load Residual UNet + diffusion schedule
    r_model_cfg = resid_cfg['residual']['model']
    resid_unet = ImprovedDiffusionUNet(
        window_size=W_r,
        feature_dim=D,
        base_channels=r_model_cfg['base_channels'],
        emb_dim=r_model_cfg['time_emb_dim'],
        n_res_blocks_per_level=r_model_cfg['n_res_blocks']
    ).to(device)

    resid_ckpt_path = resid_ckpt or os.path.join(resid_cfg['training']['ckpt_dir'], "residual_final.pt")
    full_state_r = torch.load(resid_ckpt_path, map_location=device)
    unet_state_r = {k.replace("unet.", ""): v for k, v in full_state_r.items() if k.startswith("unet.")}
    resid_unet.load_state_dict(unet_state_r)
    resid_unet.eval()

    T_r_diff = resid_cfg['diffusion']['steps']
    _, _, alpha_bars_r = cosine_beta_schedule(T_r_diff)
    alpha_bars_r = alpha_bars_r.to(device)

    # 4) Load Trend UNet + diffusion schedule
    t_model_cfg = trend_cfg['trend']['model']
    trend_unet = ImprovedDiffusionUNet(
        window_size=W_t,
        feature_dim=D,
        base_channels=t_model_cfg['base_channels'],
        emb_dim=t_model_cfg['time_emb_dim'],
        n_res_blocks_per_level=t_model_cfg['n_res_blocks']
    ).to(device)

    trend_ckpt_path = trend_ckpt or os.path.join(trend_cfg['training']['ckpt_dir'], "trend_final.pt")
    full_state_t = torch.load(trend_ckpt_path, map_location=device)
    unet_state_t = {k.replace("unet.", ""): v for k, v in full_state_t.items() if k.startswith("unet.")}
    trend_unet.load_state_dict(unet_state_t)
    trend_unet.eval()

    T_t_diff = trend_cfg['diffusion']['steps']
    _, _, alpha_bars_t = cosine_beta_schedule(T_t_diff)
    alpha_bars_t = alpha_bars_t.to(device)

    # 5) Use t = T//2
    t_r = T_r_diff // 2
    t_t = T_t_diff // 2

    # 6) Reconstruction & MSE
    rec_r = reconstruct_window(resid_unet, win_resid, alpha_bars_r, t_r, device)
    mse_r = float(np.mean((rec_r - win_resid) ** 2))

    rec_t = reconstruct_window(trend_unet, win_trend, alpha_bars_t, t_t, device)
    mse_t = float(np.mean((rec_t - win_trend) ** 2))

    # 7) Plot & save
    feature = feature_idx

    # 7a) Residual reconstruction
    t_axis_r = np.arange(W_r)
    plt.figure(figsize=(8, 4))
    plt.plot(t_axis_r, win_resid[:, feature], label="Original Residual", linewidth=2)
    plt.plot(t_axis_r, rec_r[:, feature], label=f"Reconstructed @ t={t_r}", linestyle="--")
    plt.title(f"Residual Reconstruction (middle chunk)  MSE={mse_r:.4e}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resid_recon_mid.png"))
    plt.close()

    # 7b) Trend reconstruction
    t_axis_t = np.arange(W_t)
    plt.figure(figsize=(8, 4))
    plt.plot(t_axis_t, win_trend[:, feature], label="Original Trend", linewidth=2)
    plt.plot(t_axis_t, rec_t[:, feature], label=f"Reconstructed @ t={t_t}", linestyle="--")
    plt.title(f"Trend Reconstruction (middle chunk)  MSE={mse_t:.4e}")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_recon_mid.png"))
    plt.close()

    # 8) Write MSE report
    report_path = os.path.join(out_dir, "mse_report.txt")
    with open(report_path, "w") as f:
        f.write("Residual Reconstruction (middle chunk) MSE:\n")
        f.write(f"  t={t_r:<4d}   MSE={mse_r:.6e}\n\n")
        f.write("Trend Reconstruction (middle chunk) MSE:\n")
        f.write(f"  t={t_t:<4d}   MSE={mse_t:.6e}\n")

    print(f"Evaluation complete. Outputs saved to {out_dir}.")
