
"""Bollinger Bands indicator."""

import math
import pandas as pd

def bollinger_bands( series: pd.Series, window: int = 20, num_std: float = 2.0, ) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Parameters
        series : pd.Series
            Input price series, usually closing prices.
        window : int, default=20
            Rolling window for the middle band and standard deviation.
        num_std : float, default=2.0
            Number of standard deviations for the upper and lower bands.

    Returns
        pd.DataFrame
            DataFrame containing middle, upper, and lower bands.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer")

    if window <= 0:
        raise ValueError("window must be positive")

    if isinstance(num_std, bool) or not isinstance(num_std, (int, float)):
        raise TypeError("num_std must be a number")

    if not math.isfinite(num_std) or num_std <= 0:
        raise ValueError("num_std must be positive and finite")

    middle = series.rolling(window=window, min_periods=window, ).mean()

    rolling_std = series.rolling( window=window, min_periods=window, ).std(ddof=1)

    upper = middle + num_std*rolling_std
    lower = middle - num_std*rolling_std

    return pd.DataFrame( { "middle": middle, "upper": upper, "lower": lower, }, index=series.index, )
