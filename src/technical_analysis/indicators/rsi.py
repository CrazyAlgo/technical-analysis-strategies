"""Relative Strength Index (RSI) indicator."""

import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index using Wilder's smoothing.

    Parameters
        series : pd.Series
            Input price series, usually closing prices.
        period : int, default=14
            Lookback period used to calculate RSI. The default is 14.

    Returns
        pd.Series
            RSI values between 0 and 100, indicating momentum.
            Values above 70 or below 30 are commonly interpreted as
            overbought or oversold, respectively.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError("period must be an integer")

    if period <= 0:
        raise ValueError("period must be positive")

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha = 1/period, min_periods = period, adjust = False, ).mean()

    avg_loss = losses.ewm(alpha = 1/period, min_periods = period, adjust = False, ).mean()

    relative_strength = avg_gain.div(avg_loss.where(avg_loss.ne(0)))

    result = 100 - (100/ (1 + relative_strength))

    # No losses but some gains: RSI is 100.
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)

    # No gains and no losses: RSI is undefined by convention here.
    result = result.mask((avg_loss == 0) & (avg_gain == 0), float("nan"))

    result.name = "rsi"
    return result


