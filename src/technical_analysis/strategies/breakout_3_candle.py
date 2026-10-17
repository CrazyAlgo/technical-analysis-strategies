"""BREAKOUT 3 CANDLE STRATEGY"""

import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate trading signals based on a 3-candle breakout strategy.

    A buy signal (1) is generated when the closing price breaks above
    the highest high of the previous three candles.

    A sell/short signal (-1) is generated when the closing price breaks
    below the lowest low of the previous three candles.

    A neutral signal (0) is generated otherwise.

    Parameters
        df: pd.DataFrame
        DataFrame containing 'High', 'Low' and 'Close' columns.

    Returns
        pd.DataFrame
             A copy of the input df with additional colums: 
             'three_candle_high', 'three_candle_low', and 'signal'.
    """

    required_columns = {"High", "Low", "Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError( f"DataFrame must contain these columns: {required_columns}")

    result = df.copy()

    # Highest high and lowest low of the PREVIOUS three candles
    result["three_candle_high"] = ( result["High"].shift(1).rolling(window = 3).max() )
    result["three_candle_low"] = ( result["Low"].shift(1).rolling(window = 3).min() )

    # Initialize all signals as neutral
    result["signal"] = 0

    # Bullish breakout
    result.loc[result["Close"] > result["three_candle_high"], "signal"] = 1

    # Bearish breakout
    result.loc[result["Close"] < result["three_candle_low"], "signal"] = -1

    return result
