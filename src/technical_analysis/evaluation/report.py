
"""Consolidated reporting for trading strategies"""

import numpy as np
import pandas as pd

from . import metrics
from . import trade_analysis
from . import transaction_costs


def generate_report(
    backtest_results: dict,
    benchmark_returns: pd.Series | None = None,
    trade_pnl: pd.Series | None = None,
    holding_periods: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Generate a consolidated strategy evaluation report.

    Parameters
        backtest_results: dict
        Results from backtest.run_backtest().

        benchmark_returns : pd.Series or None, default None
        Benchmark returns aligned with portfolio returns.

        trade_pnl : pd.Series or None, default None
        Realized P&L for completed trades, ordered chronologically.

        holding_periods : pd.Series or None, default None
        Holding periods aligned with trade_pnl.

        risk_free_rate : float, default 0.0
        Annualized risk-free rate expressed as a decimal.

        periods_per_year : int, default 252
        Number of return periods per year.

    Returns
        dict
        Consolidated report containing performance metrics, backtest
        summary, transaction cost analysis, and optional trade analysis.
    """

    #Validate backtest results.
    if not isinstance(backtest_results, dict):
        raise TypeError("backtest_results must be a dictionary.")

    req_keys = {
        "gross_returns",
        "net_returns",
        "turnover",
        "transaction_costs",
        "gross_equity_curve",
        "equity_curve",
        "initial_capital",
    }

    missing_keys = req_keys.difference(backtest_results)

    if missing_keys:
        raise ValueError(
            f"backtest_results is missing required keys: {sorted(missing_keys)}"
        )

    gross_returns = backtest_results["gross_returns"]
    net_returns = backtest_results["net_returns"]
    turnover = backtest_results["turnover"]
    costs = backtest_results["transaction_costs"]
    gross_equity = backtest_results["gross_equity_curve"]
    net_equity = backtest_results["equity_curve"]
    initial_capital = backtest_results["initial_capital"]

    #Validate time-series outputs and their alignment.
    series_outputs = {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "transaction_costs": costs,
        "gross_equity_curve": gross_equity,
        "equity_curve": net_equity,
    }

    for name, series in series_outputs.items():
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"backtest_results['{name}'] must be a pandas Series."
            )

        if series.empty:
            raise ValueError(f"backtest_results['{name}'] cannot be empty.")

        if series.isna().any():
            raise ValueError(
                f"backtest_results['{name}'] cannot contain missing values."
            )

        try:
            values = series.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"backtest_results['{name}'] must contain numeric values."
            ) from exc

        if not np.isfinite(values).all():
            raise ValueError(
                f"backtest_results['{name}'] must contain finite values."
            )

    reference_index = net_returns.index

    for name, series in series_outputs.items():
        if not series.index.equals(reference_index):
            raise ValueError(
                f"{name} must have the same index as net_returns."
            )

    #performance metrics.
    performance = metrics.calculate_metrics(
        returns=net_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        initial_capital=initial_capital,
    )

    #Backtest result summary
    backtest_summary = {
        "start_date": net_returns.index[0],
        "end_date": net_returns.index[-1],
        "number_of_periods": len(net_returns),
        "initial_capital": float(initial_capital),
        "final_gross_equity": float(gross_equity.iloc[-1]),
        "final_net_equity": float(net_equity.iloc[-1]),
        "total_turnover": float(turnover.sum()),
    }

    #Transaction cost summary.
    cost_summary = transaction_costs.analyze_transaction_costs(
        transaction_costs=costs,
        gross_returns=gross_returns,
        net_returns=net_returns,
    )

    #Analysis of completed trades when supplied.
    if holding_periods is not None and trade_pnl is None:
        raise ValueError(
            "trade_pnl must be supplied when holding_periods is provided."
        )

    trade_summary = None

    if trade_pnl is not None:
        trade_summary = trade_analysis.analyze_trades(
            trade_pnl=trade_pnl,
            holding_periods=holding_periods,
        )

    #Assemble report.
    return {
        "performance": performance,
        "backtest_summary": backtest_summary,
        "transaction_cost_analysis": cost_summary,
        "trade_analysis": trade_summary,
    }
