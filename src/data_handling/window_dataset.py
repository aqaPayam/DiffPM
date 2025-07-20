# in voroodi N , T , D hast N ta time series harkodoon toole T 
# ba D ta feture bad khorooji get item mishe 
# age inaro khiari poshte ham bezari
# window e idx omesh

# src/data_handling/window_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Sequence, Union


class SeriesWindowDataset(Dataset):
    """
    Deterministic “window‐of‐series” dataset.  
    Builds an index map of all possible contiguous windows (of length `window_size`) 
    across every series in the provided list.

    Each item is a dict containing:
      - 'window':     FloatTensor of shape (window_size, D_features)
      - 'start_idx':  LongTensor scalar indicating start index in that series
      - 'series_len': LongTensor scalar indicating length of the full series
    """
    def __init__(
        self,
        series_list: Union[np.ndarray, Sequence[np.ndarray]],
        window_size: int
    ):
        super().__init__()
        # Normalize input into a list of (T, D) numpy arrays
        if isinstance(series_list, np.ndarray):
            # shape (N, T, D) → list of N arrays
            self.series_list = [s for s in series_list]
        else:
            self.series_list = list(series_list)

        self.window_size = window_size

        # Build index map: each entry is (series_idx, start_idx)
        self.index_map = []
        for i, series in enumerate(self.series_list):
            T, D = series.shape
            if T < window_size:
                continue
            # windows start at 0, 1, ..., T - window_size
            for start in range(T - window_size + 1):
                self.index_map.append((i, start))

        if not self.index_map:
            raise ValueError(
                f"No series of length >= {window_size}; cannot build any windows."
            )

    def __len__(self) -> int:
        """Total number of windows across all series."""
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns the idx-th window as a dict:
          'window'     → (window_size, D) FloatTensor
          'start_idx'  → scalar LongTensor
          'series_len' → scalar LongTensor
        """
        series_idx, start = self.index_map[idx]
        series = self.series_list[series_idx]   # (T, D)
        T, D = series.shape

        window = series[start : start + self.window_size]  # (window_size, D)

        return {
            'window':     torch.from_numpy(window).float(),
            'start_idx':  torch.tensor(start, dtype=torch.long),
            'series_len': torch.tensor(T, dtype=torch.long),
        }
