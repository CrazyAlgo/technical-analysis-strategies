"""Utilities for validating financial price data."""

import pandas as pd


def validate_price_data( prices: pd.DataFrame, required_columns: tuple[str, ...] = ("Close",) ) -> pd.DataFrame:
    """Validate required columns and basic price-data integrity."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if not prices.columns.is_unique:
        raise ValueError("prices cannot contain duplicate column names.")

    if not all(isinstance(col, str) for col in prices.columns):
        raise TypeError("All column names must be strings.")

    if not required_columns:
        raise ValueError("required_columns cannot be empty.")

    missing = set(required_columns) - set(prices.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if prices.index.has_duplicates:
        raise ValueError("prices cannot contain duplicate index values.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted in chronological order.")

    for column in required_columns:
        if not pd.api.types.is_numeric_dtype(prices[column].dtype):
            raise TypeError(f"{column} must contain numeric values.")

    return prices
