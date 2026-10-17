"""SMA_CROSSOVER"""

import pandas as pd
from technical_analysis.indicators import sma


def generate_signals( df: pd.DataFrame, long_window: int, small_window: int, ) -> pd.DataFrame:
    """
    Generate signals using a simple moving average crossover.

    A buy signal (1) is generated when the fast SMA crosses
    above the slow SMA.

    A sell/short signal (-1) is generated when the fast SMA
    crosses below the slow SMA.

    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError( f"DataFrame must contain these columns: {required_columns}" )

    if small_window <= 0 or long_window <= 0:
        raise ValueError("Moving average windows must be positive.")

    if small_window >= long_window:
        raise ValueError( "small_window must be less than long_window." )

    result = df.copy()

    result["small_sma"] = sma( result["Close"], window=small_window )
    result["long_sma"] = sma( result["Close"], window=long_window )
    result["signal"] = 0

    # Bullish crossover: fast SMA crosses above slow SMA
    bullish = ( (result["small_sma"] > result["long_sma"])
        & ( result["small_sma"].shift(1) <= result["long_sma"].shift(1) ) )

    # Bearish crossover: fast SMA crosses below slow SMA
    bearish = ( (result["small_sma"] < result["long_sma"])
        & (result["small_sma"].shift(1)>= result["long_sma"].shift(1)) )

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
