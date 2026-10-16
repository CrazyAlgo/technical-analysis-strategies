"""Moving average indicators.

This module provides functions to calculate Simple Moving
Average (SMA) and Exponential Moving Average (EMA).
"""

import pandas as pd

def sma(series : pd.Series, window: int) -> pd.Series:
    """ Calculate the Simple Moving Average (SMA).

    All observations have same weights.

    Parameters
        series: pd.series
            Input price series, usually closing prices.
        window: int
            Number of obervations used to calculate the rolling average

    Returns
        pd.Series 
            Rolling simple moving average (window = int).
    
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas series")

    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer")

    if window <= 0:
        raise ValueError("window must be positive")

    return series.rolling(window = window, min_periods = window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """ Calculate the Exponential Moving Average (EMA).

    Recent observations receive greater weight than the olders ones.

    Parameters
        series : pd.Series
            Input price series, usually closing prices.
        span : int
            Span used to calculate the exponential moving average, just like the window size

    Returns
        pd.series
            Exponential moving average.

    """

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas series")
    
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError("span must be an integer")
    
    if span <= 0:
        raise ValueError("span must be positive")

    return series.ewm(span = span, adjust = False, min_periods = span).mean()

