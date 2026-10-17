"""RSI_STRATEGY"""

import pandas as pd
from technical_analysis.indicators import rsi


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals using the rsi strategy.

    A buy signal (1) is generated when RSI crosses back above 30.

    A sell/short signal (-1) is generated when RSI crosses back below 70.
    
    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain these columns: {required_columns}"
        )

    result = df.copy()

    result["rsi"] = rsi(result["Close"],)

    # Bullish threshold crossing
    bullish = ((result["rsi"] > 30) & (result["rsi"].shift(1) <= 30) )

    # Bearish threshold crossing
    bearish = ( (result["rsi"] < 70) & (result["rsi"].shift(1) >= 70) )

    result["signal"] = 0

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
