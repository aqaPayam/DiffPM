# src/data_handling/trend_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class TrendWindowDataset(Dataset):
    """
    Deterministic “window‐of‐trend” dataset.  First downsamples each series
    by keeping every `sma_window`‐th point, then builds an index_map of all
    possible contiguous windows (of length `window_size`) on the downsampled versions.
    __getitem__(idx) returns the idx-th window (no randomness).

    Each item is a dict containing:
      - 'window':     FloatTensor of shape (window_size, D_features)
      - 'start_idx':  LongTensor scalar indicating start index in downsampled series
      - 'series_len': LongTensor scalar indicating length of downsampled series
    """
    def __init__(self, series_list, window_size: int, sma_window: int = 1):
        super().__init__()
        # Normalize input into a list of (T, D) numpy arrays
        if isinstance(series_list, np.ndarray):
            self.orig_series = [s for s in series_list]
        else:
            self.orig_series = list(series_list)

        self.window_size = window_size
        self.sma_window = max(1, sma_window)

        # Build downsampled series list
        self.ds_series_list = []
        for arr in self.orig_series:
            if self.sma_window > 1:
                ds = arr[::self.sma_window]
            else:
                ds = arr.copy()
            T_ds, D = ds.shape
            if T_ds < window_size:
                continue
            self.ds_series_list.append(ds)

        if not self.ds_series_list:
            raise ValueError(
                "No series remain long enough after downsampling; reduce sma_window or window_size."
            )

        # Build index_map: each entry is (ds_idx, start_idx)
        self.index_map = []
        for i, ds in enumerate(self.ds_series_list):
            T_ds, D = ds.shape
            n_windows = T_ds - window_size + 1
            for start in range(n_windows):
                self.index_map.append((i, start))

    def __len__(self):
        # Total number of possible windows across all downsampled series
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1) Lookup which downsampled series and start‐index
        ds_idx, start = self.index_map[idx]
        ds = self.ds_series_list[ds_idx]      # (T_ds, D)
        T_ds, D = ds.shape

        # 2) Slice out the window
        window = ds[start : start + self.window_size]  # (window_size, D)

        return {
            'window':     torch.from_numpy(window).float(),
            'start_idx':  torch.tensor(start, dtype=torch.long),
            'series_len': torch.tensor(T_ds, dtype=torch.long),
        }
