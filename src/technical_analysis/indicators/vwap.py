"""Volume Weighted Average Price (VWAP) indicator."""

import pandas as pd

def vwap (high:pd.Series, low:pd.Series , close:pd.Series, volume:pd.Series) -> pd.Series:
    """Calculate cumulative Volume Weighted Average Price.

     typical_price = (High + Low + Close) / 3. 
     The typical_price represents the trading period using its high, low, and closing prices. 
     The Open is excluded because this conventional VWAP formulation uses the typical_price.

    Parameters
        high : pd.Series
            High prices.
        low : pd.Series
            Low prices.
        close : pd.Series
            Closing prices.
        volume : pd.Series
            Trading volume.

    Returns
        pd.Series
            Cumulative VWAP over the supplied data.
    """

    inputs = {
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    for name, series in inputs.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series")

    if not all( series.index.equals(close.index) for series in inputs.values() ):
        raise ValueError("All input Series must have identical indices")

    if (volume.dropna() < 0).any():
        raise ValueError("volume cannot contain negative values")

    typical_price = (high+ low + close)/3

    price_volume = typical_price*volume

    cumulative_volume = volume.cumsum()
    cumulative_price_volume = price_volume.cumsum()

    # VWAP = cum_P*V/ cum_V
    result = cumulative_price_volume.div(cumulative_volume.where(cumulative_volume.ne(0)))

    result.name = "vwap"
    return result

