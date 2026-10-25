
"""Reusable validation utilities."""

import numbers

import numpy as np
import pandas as pd


def validate_series( series: pd.Series, name: str = "series") -> pd.Series:
    """Validate a non-empty Series containing finite numeric values."""

    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    if series.empty:
        raise ValueError(f"{name} cannot be empty.")

    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{name} must contain numeric values.")

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values.")

    return series


def validate_positive_number( value: numbers.Real, name: str = "value") -> float:
    """Validate a positive, finite real number."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Real
    ):
        raise TypeError(f"{name} must be a real number.")

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return float(value)


def validate_positive_integer( value: int, name: str = "value") -> int:
    """Validate a positive integer."""

    if isinstance(value, (bool, np.bool_)) or not isinstance( value, (int, np.integer) ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return int(value)
