import numpy as np

def split_trend_residual(data: np.ndarray):
    """
    Decompose multivariate series into trend (low‐frequency) and residual components,
    using only a moving‐average for trend.

    Args
    ----
    data : np.ndarray
        Shape (N, T, D).  N series, T timesteps, D features.

    Returns
    -------
    trend_list : List[np.ndarray]  # each (T, D)
    resid_list : List[np.ndarray]  # each (T, D)
    """
    trend_list, resid_list = [], []

    for series in data:  # series has shape (T, D)
        T, D = series.shape
        trend = np.zeros_like(series)
        resid = np.zeros_like(series)

        # Determine window size once per-series (could also do per-feature, but typically same T)
        win = max(3, T // 20)  # heuristic: at least 3 points, or 5% of length

        for d in range(D):
            y = series[:, d]  # univariate timeseries of length T

            # Compute moving average (low‐frequency trend)
            mov_avg = np.convolve(y, np.ones(win) / win, mode="same")
            trend[:, d] = mov_avg
            resid[:, d] = y - mov_avg

        trend_list.append(trend)
        resid_list.append(resid)

    return trend_list, resid_list
