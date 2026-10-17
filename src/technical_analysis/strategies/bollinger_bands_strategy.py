"""BOLLINGER_BANDS_STRATEGY"""

import pandas as pd
from technical_analysis.indicators import bollinger_bands


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals using a Bollinger Bands strategy.

    A buy signal (1) is generated when close crosses back above the lower band

    A sell/short signal (-1) is generated when close crosses back below the upper band.
    
    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain these columns: {required_columns}"
        )

    result = df.copy()

    bands = bollinger_bands(result["Close"])
    result = result.join(bands)

    # Bullish threshold crossing
    bullish = ( (result["Close"] > result["lower"])  & (result["Close"].shift(1) <= result["lower"].shift(1)) )

    # Bearish threshold crossing
    bearish = ( (result["Close"] < result["upper"])  & (result["Close"].shift(1) >= result["upper"].shift(1)) )

    result["signal"] = 0

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
