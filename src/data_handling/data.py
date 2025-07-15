import os
import gzip
import numpy as np
import pandas as pd
from darts.datasets import WeatherDataset

def load_dataset(
    dataset_name: str,
    data_dir: str = '/workspace/BatterySOH/project_root/data',
    batch_size: int = 1
) -> np.ndarray:
    """
    Load a multivariate time series dataset by name from a specified directory and reshape into batches.

    Parameters
    ----------
    dataset_name : str
        One of: 'solar', 'electricity', 'traffic', 'exchange',
        'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'stock', 'pems', 'wind', ...
    data_dir : str
        Path to the directory containing the dataset files (not used for 'weather').
    batch_size : int
        Number of batches to split the time series into. The time dimension
        must be divisible by batch_size.

    Returns
    -------
    data : np.ndarray
        Array of shape (batch_size, length, feature_num), where
        length = total_time_steps // batch_size and feature_num is the number of series.
    """
    # Handle 'weather' via Darts directly
    if dataset_name == 'weather':
        ts = WeatherDataset().load()
        df = ts.pd_dataframe()
        raw_data = df.values
    else:
        # Define opener/loader for file-based datasets
        opener = None
        loader = None

        if dataset_name == 'solar':
            filename = 'solar_AL.txt.gz'
            opener = lambda fp: gzip.open(fp, mode='rt')
            loader = lambda f: np.loadtxt(f, delimiter=',')
        elif dataset_name == 'electricity':
            filename = 'electricity.txt.gz'
            opener = lambda fp: gzip.open(fp, mode='rt')
            loader = lambda f: np.loadtxt(f, delimiter=',')
        elif dataset_name == 'traffic':
            filename = 'traffic.txt.gz'
            opener = lambda fp: gzip.open(fp, mode='rt')
            loader = lambda f: np.loadtxt(f, delimiter=',')
        elif dataset_name == 'exchange':
            filename = 'exchange_rate.txt.gz'
            opener = lambda fp: gzip.open(fp, mode='rt')
            loader = lambda f: np.loadtxt(f, delimiter=',')
        elif dataset_name in {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}:
            csv_map = {'ETTh1': 'ETTh1.csv', 'ETTh2': 'ETTh2.csv', 'ETTm1': 'ETTm1.csv', 'ETTm2': 'ETTm2.csv'}
            filename = csv_map[dataset_name]
            def loader(fp):
                df = pd.read_csv(fp, parse_dates=[0], index_col=0)
                return df.values
        elif dataset_name == 'stock':
            filename = 'stock_data.csv'
            def loader(fp):
                df = pd.read_csv(fp)
                return df.values
        elif dataset_name.lower() == 'pems':
            filename = 'PEMS-BAY.csv'
            def loader(fp):
                df = pd.read_csv(fp, parse_dates=[0], index_col=0)
                return df.values
        elif dataset_name == 'wind':
            filename = 'W_15.csv'
            def loader(fp):
                df = pd.read_csv(fp, parse_dates=[0], index_col=0)
                return df.values
            
            
        elif dataset_name == 'battery':
            # Load the third column of 2C_battery-1.csv as a (T,1) series
            filename = '2C_battery-1.csv'
            def loader(fp):
                df = pd.read_csv(fp)
                # extract third column (zero-based idx 2), make shape (T,1)
                col = df.iloc[:, 2].to_numpy().reshape(-1, 1)
                return col

            
            
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        filepath = os.path.join(data_dir, filename)
        if opener is not None:
            with opener(filepath) as f:
                raw_data = loader(f)
        else:
            raw_data = loader(filepath)

    # Convert to numpy array
    data_arr = np.asarray(raw_data)
    if data_arr.ndim != 2:
        raise ValueError(f"Loaded data must be 2D, got shape {data_arr.shape}")

    time_steps, feature_num = data_arr.shape
    if time_steps % batch_size != 0:
        raise ValueError(
            f"Time dimension ({time_steps}) is not divisible by batch_size ({batch_size})."
        )

    length = time_steps // batch_size
    data = data_arr[: batch_size * length]
    data = data.reshape(batch_size, length, feature_num)

    return data