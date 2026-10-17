"""EMA_CROSSOVER"""

import pandas as pd
from technical_analysis.indicators import ema


def generate_signals( df: pd.DataFrame, long_span: int, small_span: int, ) -> pd.DataFrame:
    """
    Generate signals using an exponential moving average crossover.

    A buy signal (1) is generated when the fast EMA crosses
    above the slow EMA.

    A sell/short signal (-1) is generated when the fast EMA
    crosses below the slow EMA.

    A neutral signal (0) is generated otherwise.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError( f"DataFrame must contain these columns: {required_columns}" )

    if small_span <= 0 or long_span <= 0:
        raise ValueError("EMA_spans must be positive.")

    if small_span >= long_span:
        raise ValueError( "small_span must be less than long_span." )

    result = df.copy()

    result["small_ema"] = ema( result["Close"],span=small_span )
    result["long_ema"] = ema( result["Close"],span=long_span )
    result["signal"] = 0

    # Bullish crossover: fast EMA crosses above slow EMA
    bullish = ( (result["small_ema"] > result["long_ema"])
        & ( result["small_ema"].shift(1) <= result["long_ema"].shift(1) ) )

    # Bearish crossover: fast EMA crosses below slow EMA
    bearish = ( (result["small_ema"] < result["long_ema"])
        & (result["small_ema"].shift(1)>= result["long_ema"].shift(1)) )

    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result
