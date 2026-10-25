
"""
Evaluation package for backtesting and evaluating trading strategies.
Provides modules for backtesting, performance metrics, trade analysis,
transaction costs, visualizations, heatmaps, and report generation.
"""

from . import backtest
from . import heatmaps
from . import metrics
from . import plots
from . import report
from . import trade_analysis
from . import transaction_costs


__all__ = [
    "backtest",
    "heatmaps",
    "metrics",
    "plots",
    "report",
    "trade_analysis",
    "transaction_costs"
]