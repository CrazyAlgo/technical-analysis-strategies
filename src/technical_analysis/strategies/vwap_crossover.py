"""VWAP_CROSSOVER"""

import pandas as pd
from technical_analysis.indicators import vwap


def generate_signals( df: pd.DataFrame, ) -> pd.DataFrame:
    """
    Generate signals using a vwap crossover .

    A buy signal (1) is generated when closing price crosses
    above the vwap.

    A sell/short signal (-1) is generated when the closing price
    crosses below the vwap.

    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"High", "Low", "Close", "Volume"}

    if not required_columns.issubset(df.columns):
        raise ValueError( f"DataFrame must contain these columns: {required_columns}" )

    result = df.copy()

    result["vwap"] = vwap( result["High"], result["Low"], result["Close"], result["Volume"])
    result["signal"] = 0

    # Bullish crossover: closing price crosses above the VWAP
    bullish = ( (result["Close"] > result["vwap"])
        & ( result["Close"].shift(1) <= result["vwap"].shift(1) ) )

    # Bearish crossover: closing price crosses below the VWAP
    bearish = ( (result["Close"] < result["vwap"])
        & (result["Close"].shift(1)>= result["vwap"].shift(1)) )

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
