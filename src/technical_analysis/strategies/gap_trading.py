"""GAP_TRADING_STRATEGY"""

import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals using a Gap Trading strategy.

    A buy signal (1) is generated when today's low is greater than yesterday's high.

    A sell/short signal (-1) is generated when today's high is lower than yesterday's low.
    
    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"High", "Low"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain these columns: {required_columns}"
        )

    result = df.copy()

    # Gap Up
    bullish = ( (result["Low"] > result["High"].shift(1)) )

    # Gap Down
    bearish = ( (result["High"] < result["Low"].shift(1)) )

    result["signal"] = 0

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
