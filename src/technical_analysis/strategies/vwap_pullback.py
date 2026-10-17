"""VWAP_PULLBACK"""

import pandas as pd
from technical_analysis.indicators import vwap


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals using a VWAP pullback strategy.

    Buy (1): At least 4 of the previous 5 closes were above VWAP,
    and the current candle touches VWAP and closes bullishly above it.

    Sell/short (-1): At least 4 of the previous 5 closes were below VWAP,
    and the current candle touches VWAP and closes bearishly below it.

    Neutral (0): No qualifying setup.
    """

    required_columns = {"Open", "High", "Low", "Close", "Volume"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain these columns: {required_columns}"
        )

    result = df.copy()

    result["vwap"] = vwap( result["High"], result["Low"], result["Close"], result["Volume"], )

    # Determine the prior trend, excluding the current candle
    above_vwap = result["Close"] > result["vwap"]
    below_vwap = result["Close"] < result["vwap"]

    trend_up = ( above_vwap.shift(1).rolling(window=5).sum() >= 4 )

    trend_down = ( below_vwap.shift(1).rolling(window=5).sum() >= 4 )

    # Pullback and bullish confirmation
    bullish = ( trend_up
                    & (result["Low"] <= result["vwap"])
                    & (result["Close"] > result["vwap"])
                    & (result["Close"] > result["Open"])
                )

    # Pullback and bearish confirmation
    bearish = ( trend_down
                    & (result["High"] >= result["vwap"])
                    & (result["Close"] < result["vwap"])
                    & (result["Close"] < result["Open"])
                )

    result["signal"] = 0

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
