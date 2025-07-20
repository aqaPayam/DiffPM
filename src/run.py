import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from data_handling.decomposition import (
    split_trend_residual,
    downsample_trend,
)
from data_handling.window_dataset import SeriesWindowDataset
from training.train import train_diffusion_model


def run(
    data_npy: str,
    *,
    ma_window_size: int,
    window_size: int,
    feature_dim: int,
    emb_dim: int,
    base_channels: int,
    num_res_blocks: int,
    num_timesteps: int,
    schedule_s: float,
    stat_loss_weight: float,
    lr: float,
    batch_size: int,
    num_epochs: int,
    device: str = "cuda",
    log_root: str = "logs",
) -> Tuple[str, str]:
    """
    Full pipeline:
      1. Load raw (N, T, D) data from `data_npy`.
      2. Decompose into trend/residual using MA window `ma_window_size`.
      3. Downsample trend by exactly that `ma_window_size`.
      4. Plot the raw series, trend, residual, and downsampled trend for inspection.
      5. Build two SeriesWindowDatasets (length `window_size`):
           - trend_dataset on downsampled trends
           - resid_dataset on raw residuals
      6. Train two DiffusionModels (one per dataset) with identical hyperparameters
         except for their data sources.

    Returns
    -------
    (trend_log_dir, resid_log_dir)
    """
    # 1) Load raw data
    arr = np.load(data_npy)  # shape (N, T, D)

    # 2) Decompose
    trends, resids = split_trend_residual(arr, window_size=ma_window_size)
    # Ensure numpy arrays for indexing
    trends = np.array(trends)    # shape (N, T, D)
    resids = np.array(resids)    # shape (N, T, D)

    # 3) Downsample trend
    trends_ds = downsample_trend(trends, window_size=ma_window_size)
    trends_ds = np.array(trends_ds)  # shape (N, T//ma_window_size, D)

    # 4) Plot the arr, trends, resids and trends_ds
    # Use the first series (index 0) and first feature (index 0) for visualization
    sample_idx = 0
    feature_idx = 0
    raw = arr[sample_idx, :, feature_idx]
    trend = trends[sample_idx, :, feature_idx]
    resid = resids[sample_idx, :, feature_idx]
    trend_ds = trends_ds[sample_idx, :, feature_idx]

    # Create output directory for plots
    plot_dir = os.path.join(log_root, "decomposition_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    axes[0].plot(raw)
    axes[0].set_title(f"Raw Series (sample={sample_idx}, feature={feature_idx})")
    axes[1].plot(trend)
    axes[1].set_title(f"Trend (MA window={ma_window_size})")
    axes[2].plot(resid)
    axes[2].set_title("Residual")
    axes[3].plot(trend_ds)
    axes[3].set_title("Downsampled Trend")
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, "decomposition.png")
    fig.savefig(plot_path)
    plt.close(fig)

    print(f"Decomposition plot saved to {plot_path}")

    # 5) Build datasets
    trend_dataset = SeriesWindowDataset(trends_ds, window_size=window_size)
    resid_dataset = SeriesWindowDataset(resids,     window_size=window_size)

    # 6) Train on trend
    trend_log_dir = train_diffusion_model(
        trend_dataset,
        window_size=window_size,
        feature_dim=feature_dim,
        emb_dim=emb_dim,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        num_timesteps=num_timesteps,
        schedule_s=schedule_s,
        stat_loss_weight=stat_loss_weight,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        log_root=log_root,
    )

    # 7) Train on residual
    resid_log_dir = train_diffusion_model(
        resid_dataset,
        window_size=window_size,
        feature_dim=feature_dim,
        emb_dim=emb_dim,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
        num_timesteps=num_timesteps,
        schedule_s=schedule_s,
        stat_loss_weight=stat_loss_weight,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        log_root=log_root,
    )

    return trend_log_dir, resid_log_dir
