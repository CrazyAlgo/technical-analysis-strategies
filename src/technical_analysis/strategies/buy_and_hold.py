"""BUY_AND_HOLD"""

import numpy as np
import pandas as pd

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate buy-and-hold benchmark returns.
    Assumes the asset is held throughout the testing period.
    """

    required_columns = {"Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain these columns: {required_columns}"
        )

    result = df.copy()

    # Daily log returns
    result["log_returns"] = np.log(
        result["Close"] / result["Close"].shift(1)
    )

    # Cumulative log returns
    result["cumulative_log_returns"] = (
        result["log_returns"].fillna(0).cumsum()
    )

    # Convert cumulative log returns to cumulative simple returns
    result["cumulative_returns"] = (
        np.exp(result["cumulative_log_returns"]) - 1
    )

    return result