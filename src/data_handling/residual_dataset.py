# src/data_handling/residual_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class ResidualWindowDataset(Dataset):
    """
    Deterministic “window‐of‐residuals” dataset.  Builds an index_map of
    all possible contiguous windows (of length `window_size`) across all series.
    __getitem__(idx) returns the idx-th window (no randomness).
    
    Each item is a dict containing:
      - 'window':     FloatTensor of shape (window_size, D_features)
      - 'start_idx':  LongTensor scalar indicating start index in that series
      - 'series_len': LongTensor scalar indicating length of the series
    """
    def __init__(self, series_list, window_size: int):
        super().__init__()
        # Normalize input into a list of (T, D) numpy arrays
        if isinstance(series_list, np.ndarray):
            # shape (N, T, D) → list of N arrays
            self.series_list = [s for s in series_list]
        else:
            self.series_list = list(series_list)

        self.window_size = window_size

        # Build index_map: each entry is (series_idx, start_idx)
        self.index_map = []
        for i, series in enumerate(self.series_list):
            T, D = series.shape
            if T < window_size:
                continue
            n_windows = T - window_size + 1
            for start in range(n_windows):
                self.index_map.append((i, start))

        if not self.index_map:
            raise ValueError("No series of length >= window_size; cannot build any windows.")

    def __len__(self):
        # Total number of possible windows across all series
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1) Lookup which series and start‐index
        series_idx, start = self.index_map[idx]
        series = self.series_list[series_idx]   # shape (T, D)
        T, D = series.shape

        # 2) Slice out the window
        window = series[start : start + self.window_size]  # (window_size, D)

        return {
            'window':     torch.from_numpy(window).float(),
            'start_idx':  torch.tensor(start, dtype=torch.long),
            'series_len': torch.tensor(T, dtype=torch.long),
        }
