"""PRICE_VS_MA_CROSSOVER"""

import pandas as pd
from technical_analysis.indicators import sma


def generate_signals( df: pd.DataFrame, window: int, ) -> pd.DataFrame:
    """
    Generate signals using a price vs moving average crossover .

    A buy signal (1) is generated when closing price crosses
    above the SMA.

    A sell/short signal (-1) is generated when the closing price
    crosses below the SMA.

    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError( f"DataFrame must contain these columns: {required_columns}" )

    if window <= 0:
        raise ValueError("MA window must be positive.")

    result = df.copy()

    result["ma"] = sma( result["Close"], window = window )
    result["signal"] = 0

    # Bullish crossover: closing price crosses above the MA
    bullish = ( (result["Close"] > result["ma"])
        & ( result["Close"].shift(1) <= result["ma"].shift(1) ) )

    # Bearish crossover: closing price crosses below the MA
    bearish = ( (result["Close"] < result["ma"])
        & (result["Close"].shift(1)>= result["ma"].shift(1)) )

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
