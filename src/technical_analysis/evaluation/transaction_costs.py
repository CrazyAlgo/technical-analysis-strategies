"""Transaction cost analysis for trading strategies."""

import numpy as np
import pandas as pd

from technical_analysis.evaluation.metrics import gross_return, net_return


def analyze_transaction_costs( transaction_costs: pd.Series, gross_returns: pd.Series, net_returns: pd.Series ) -> dict[str, float]:
    """Calculate transaction cost totals and their impact on returns.

    Parameters
        transaction_costs : pd.Series
            Transaction costs for each period, expressed in consistent units.
        gross_returns : pd.Series
            Portfolio returns before transaction costs.
        net_returns : pd.Series
            Portfolio returns after transaction costs.

    Returns
        dict[str, float]
            Dictionary containing transaction cost analysis metrics.
    """
    if not isinstance(transaction_costs, pd.Series):
        raise TypeError("transaction_costs must be a pandas Series.")

    if not isinstance(gross_returns, pd.Series):
        raise TypeError("gross_returns must be a pandas Series.")

    if not isinstance(net_returns, pd.Series):
        raise TypeError("net_returns must be a pandas Series.")

    if not (
        transaction_costs.index.equals(gross_returns.index)
        and transaction_costs.index.equals(net_returns.index)
    ):
        raise ValueError("All input series must have identical indices.")

    metrics: dict[str, float] = {
        "total_transaction_costs": total_transaction_costs( transaction_costs),
        "transaction_cost_drag": transaction_cost_drag( gross_returns, net_returns)
    }

    return metrics


def total_transaction_costs(transaction_costs: pd.Series) -> float:
    """Calculate total transaction costs in consistent units."""

    if not isinstance(transaction_costs, pd.Series):
        raise TypeError("transaction_costs must be a pandas Series.")

    if transaction_costs.empty:
        raise ValueError("transaction_costs cannot be empty.")

    if transaction_costs.isna().any():
        raise ValueError("transaction_costs cannot contain missing values.")

    costs = transaction_costs.to_numpy(dtype=float)

    if not np.isfinite(costs).all():
        raise ValueError("transaction_costs must be finite.")

    if (costs < 0).any():
        raise ValueError("transaction_costs cannot be negative.")

    return float(costs.sum())


def transaction_cost_drag( gross_returns: pd.Series, net_returns: pd.Series ) -> float:
    """Calculate cumulative return lost to transaction costs."""

    if not isinstance(gross_returns, pd.Series):
        raise TypeError("gross_returns must be a pandas Series.")

    if not isinstance(net_returns, pd.Series):
        raise TypeError("net_returns must be a pandas Series.")

    if not gross_returns.index.equals(net_returns.index):
        raise ValueError( "gross_returns and net_returns must have matching indexes.")

    gross = gross_return(gross_returns)
    net = net_return(net_returns)

    return float(gross - net)

