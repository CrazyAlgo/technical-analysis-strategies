"""Trade-level analysis for multi-asset trading strategies."""

import numpy as np 
import pandas as pd

def validate_trade_pnl(trade_pnl: pd.Series) -> pd.Series:
    """Validate realized profit and loss for completed trades."""

    if not isinstance(trade_pnl, pd.Series):
        raise TypeError("trade_pnl must be a pandas Series.")

    if trade_pnl.empty:
        raise ValueError("trade_pnl cannot be empty.")

    if trade_pnl.isna().any():
        raise ValueError("trade_pnl cannot contain missing value(s).")

    values = trade_pnl.to_numpy(dtype = float)

    if not np.isfinite(values).all():
        raise ValueError("trade_pnl must contain only finite values.")

    return pd.Series(values, index = trade_pnl.index)


def validate_holding_periods(holding_periods: pd.Series, trade_pnl: pd.Series) -> pd.Series:
    """Validate holding periods for completed trades."""

    if not isinstance(holding_periods, pd.Series):
        raise TypeError("holding_periods must be a pandas Series.")

    if holding_periods.empty:
        raise ValueError("holding_periods cannot be  empty.")

    if holding_periods.isna().any():
        raise ValueError("holding_periods cannot contain missing values.")

    values = holding_periods.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("holding_periods must contain only finite values.")

    if (values < 0).any():
        raise ValueError("holding_periods cannot be negative.")

    if not holding_periods.index.equals(trade_pnl.index):
        raise ValueError(
            "holding_periods and trade_pnl must have identical indices."
        )

    return pd.Series(values, index=holding_periods.index)


def analyze_trades( trade_pnl: pd.Series, holding_periods: pd.Series | None = None,) -> dict[str, float | int]:
    """Calculate all applicable trade-level performance metrics.

    Parameters
        trade_pnl : pd.Series
            Realized P&L for completed trades, ordered chronologically.
        holding_periods : pd.Series or None, default None
            Holding period for each completed trade. Must have the same
            index as trade_pnl.

    Returns
        dict[str, float | int]
            Dictionary containing trade-level performance metrics.
    """

    # Validate inputs once before calculating the metrics.
    trade_pnl = validate_trade_pnl(trade_pnl)

    if holding_periods is not None:
        holding_periods = validate_holding_periods( holding_periods, trade_pnl )

    metrics: dict[str, float | int] = {
        "total_trades": total_trades(trade_pnl),
        "win_rate": win_rate(trade_pnl),
        "loss_rate": loss_rate(trade_pnl),
        "average_winning_trade": average_winning_trade(trade_pnl),
        "average_losing_trade": average_losing_trade(trade_pnl),
        "largest_win": largest_win(trade_pnl),
        "largest_loss": largest_loss(trade_pnl),
        "profit_factor": profit_factor(trade_pnl),
        "expectancy_per_trade": expectancy_per_trade(trade_pnl),
        "average_win_loss_ratio": average_win_loss_ratio(trade_pnl),
        "maximum_consecutive_wins": maximum_consecutive_wins(trade_pnl),
        "maximum_consecutive_losses": maximum_consecutive_losses(trade_pnl),
    }

    if holding_periods is not None:
        metrics["average_holding_period"] = average_holding_period( trade_pnl, holding_periods )

    return metrics


def total_trades(trade_pnl: pd.Series) -> int: 
    """Calculate the total number of completed trades."""
    trade_pnl = validate_trade_pnl(trade_pnl)

    return int(len(trade_pnl))


def win_rate(trade_pnl: pd.Series) -> float: 
    """ Calculate  the percentage of profitable trades."""

    trade_pnl = validate_trade_pnl(trade_pnl)

    return float((trade_pnl>0).mean()*100)


def loss_rate(trade_pnl: pd.Series) -> float: 
    """ Calculate  the percentage of loosing trades."""
    
    trade_pnl = validate_trade_pnl(trade_pnl)
    
    return float((trade_pnl<0).mean()*100)


def average_winning_trade(trade_pnl:pd.Series) -> float: 
    """Calculate the average profit of winning trades."""

    trade_pnl = validate_trade_pnl(trade_pnl)
    winning_trades = trade_pnl[trade_pnl>0]

    if winning_trades.empty:
        return float("nan")

    return float(winning_trades.mean())


def average_losing_trade(trade_pnl : pd.Series) -> float: 
    """Calculate the average pnl of losing trades."""
    
    trade_pnl = validate_trade_pnl(trade_pnl)
    losing_trades = trade_pnl[trade_pnl<0]
    
    if losing_trades.empty:
       return float("nan")
    
    return float(losing_trades.mean())


def largest_win(trade_pnl : pd.Series) -> float: 
    """Calculate the largest realized winning trade."""
    
    trade_pnl = validate_trade_pnl(trade_pnl)
    winning_trades = trade_pnl[trade_pnl>0]
    
    if winning_trades.empty:
        return float("nan")

    return float(winning_trades.max())


def largest_loss(trade_pnl : pd.Series) -> float: 
    """Calculate the largest realized loss, as a negative value."""

    trade_pnl = validate_trade_pnl(trade_pnl)
    losing_trades = trade_pnl[trade_pnl < 0]

    if losing_trades.empty:
        return float("nan")

    return float(losing_trades.min())


def profit_factor(trade_pnl : pd.Series) -> float: 
    """Calculate gross profits divided by absolute gross losses."""

    trade_pnl = validate_trade_pnl(trade_pnl)

    gross_profit = float(trade_pnl[trade_pnl > 0].sum())
    gross_loss = float(trade_pnl[trade_pnl < 0].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")

    return float(gross_profit/abs(gross_loss))


def expectancy_per_trade(trade_pnl: pd.Series) -> float: 
    """Calculate the average realized P&L per completed trade."""

    trade_pnl = validate_trade_pnl(trade_pnl)

    return float(trade_pnl.mean())


def average_holding_period(trade_pnl: pd.Series, holding_periods: pd.Series) -> float: 
    """Calculate the average holding period of completed trades."""

    trade_pnl = validate_trade_pnl(trade_pnl)
    holding_periods = validate_holding_periods(holding_periods, trade_pnl)

    return float(holding_periods.mean())


def average_win_loss_ratio(trade_pnl: pd.Series) -> float: 
    """Calculate average winning P&L divided by average loss magnitude."""

    trade_pnl = validate_trade_pnl(trade_pnl)

    winning_trades = trade_pnl[trade_pnl > 0]
    losing_trades = trade_pnl[trade_pnl < 0]

    if winning_trades.empty or losing_trades.empty:
        return float("nan")

    average_win = float(winning_trades.mean())
    average_loss = float(-losing_trades.mean())

    return float(average_win/average_loss)


def maximum_consecutive_wins(trade_pnl: pd.Series) -> int: 
    """Calculate the maximum consecutive winning-trade streak."""

    trade_pnl = validate_trade_pnl(trade_pnl)

    longest_streak = 0
    current_streak = 0

    for pnl in trade_pnl:
        if pnl > 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else :
            current_streak = 0

    return longest_streak
        

def maximum_consecutive_losses(trade_pnl: pd.Series) -> int: 
    """Calculate the maximum consecutive losing-trade streak."""
    
    trade_pnl = validate_trade_pnl(trade_pnl)
    
    longest_streak = 0
    current_streak = 0
    
    for pnl in trade_pnl:
        if pnl < 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else :
               current_streak = 0
    
    return longest_streak