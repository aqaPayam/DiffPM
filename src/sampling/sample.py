# src/sampling/sample.py

import math
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

from data_handling.decomposition import upsample_trend


def reconstruct_from_windows(
    windows: List[np.ndarray],
    series_length: int,
    window_size: int
) -> np.ndarray:
    """
    Given overlapping windows (each shape (W, D)), reconstruct the full
    series (series_length, D) by summing and averaging overlaps.
    """
    D = windows[0].shape[1]
    acc = np.zeros((series_length, D), dtype=np.float32)
    count = np.zeros((series_length, D), dtype=np.int32)

    for start, w in enumerate(windows):
        end = start + window_size
        acc[start:end] += w
        count[start:end] += 1

    count[count == 0] = 1
    return acc / count


@torch.no_grad()
def sample_full_series(
    trend_model: torch.nn.Module,
    resid_model: torch.nn.Module,
    original_length: int,
    ma_window_size: int,
    window_size: int,
    device: str = "cpu",
    plot: bool = False
) -> np.ndarray:
    """
    1) Slide a window of length `window_size` over residual grid,
       sample each window from resid_model.sample(...)
    2) On the downsampled grid of length ceil(original_length/ma_window_size),
       slide same-length windows, sample each window from trend_model.sample(...)
    3) Reconstruct each (residual & downsampled-trend) full series by overlap-average
    4) Upsample the reconstructed downsampled-trend back to original_length
    5) Sum trend + residual → final series (original_length, D)
    If plot=True, also plot residual_series, trend before upsampling, and trend after upsampling.
    """
    resid_windows: List[np.ndarray] = []
    for start in range(0, original_length - window_size + 1):
        si = torch.tensor([start], dtype=torch.long, device=device)
        sl = torch.tensor([original_length], dtype=torch.long, device=device)
        xw = resid_model.sample(si, sl)  # (1, W, D)
        resid_windows.append(xw.squeeze(0).cpu().numpy())

    ds_length = math.ceil(original_length / ma_window_size)
    trend_windows: List[np.ndarray] = []
    for start in range(0, ds_length - window_size + 1):
        si = torch.tensor([start], dtype=torch.long, device=device)
        sl = torch.tensor([ds_length], dtype=torch.long, device=device)
        xw = trend_model.sample(si, sl)  # (1, W, D)
        trend_windows.append(xw.squeeze(0).cpu().numpy())

    resid_series = reconstruct_from_windows(resid_windows, original_length, window_size)
    trend_ds = reconstruct_from_windows(trend_windows, ds_length, window_size)

    trend_us = upsample_trend(
        [trend_ds],
        original_length=original_length,
        window_size=ma_window_size,
        method="sinc"
    )[0]  # (original_length, D)

    final_series = trend_us + resid_series

    if plot:
        # Plot reconstructed residual
        plt.figure(figsize=(10, 3))
        plt.plot(resid_series.squeeze(), label="Residual Series")
        plt.title("Reconstructed Residual Series")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot trend before and after upsampling
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        ds_grid = np.arange(ds_length)
        orig_grid = np.arange(original_length)

        axes[0].plot(ds_grid, trend_ds.squeeze(), label="Downsampled Trend")
        axes[0].set_title("Trend Before Upsampling (Downsampled)")
        axes[0].set_xlabel("Downsampled Index")
        axes[0].set_ylabel("Value")
        axes[0].legend()

        axes[1].plot(orig_grid, trend_us.squeeze(), label="Upsampled Trend")
        axes[1].set_title("Trend After Upsampling")
        axes[1].set_xlabel("Time Index")
        axes[1].set_ylabel("Value")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return final_series
