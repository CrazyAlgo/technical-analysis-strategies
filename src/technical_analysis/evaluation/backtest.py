"""Multi-asset portfolio backtesting engine."""

import numpy as np 
import pandas as pd 

def run_backtest( prices: pd.DataFrame, 
                 positions: pd.DataFrame, 
                 initial_capital: float = 100_000.0, 
                 transaction_cost_rate: float = 0.001
                 ) -> dict[str, pd.DataFrame | pd.Series | float]:
    """Backtest a multi-asset portfolio using target weights.
    
    Parameters
        prices: pd.DataFrame
            positive asset prices. Rows are timestamps and columns are assets.
        
        positions: pd.DataFrame
            Target portfolio weights. Rows are timestamps and columns are assets.
            Weights observed at time t are held from the next bar onwards.
            Negative weights represent short positions.

        initial_capital : float, default = 100_000.0
            Starting portfolio value.

        transaction_cost_rate : float, default 0.001
            Proportional transaction cost per unit of portfolio turnover

    Returns
        dict
            Asset returns, held positions, portfolio returns, turnover, transaction costs,
            and gross and net equity curves.
    """

    #Validate input types.
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if positions.empty:
        raise ValueError("positions cannot be empty.")

    #validate numeric data
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices.dtypes):
        raise TypeError("prices must contain only numeric columns.")

    if not all(pd.api.types.is_numeric_dtype(dtype)for dtype in positions.dtypes):
        raise TypeError("positions must contain only numeric columns.")

    if prices.isna().any().any():
        raise ValueError("prices cannot contain missing values.")

    if positions.isna().any().any():
        raise ValueError("positions cannot contain missing values")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("prices must contain only finite values.") 

    if not np.isfinite(positions.to_numpy(dtype=float)).all():
        raise ValueError("positions must contain only finite values.")

    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive.")

    #validate indices and asset columns.
    if not prices.index.equals(positions.index):
        raise ValueError("prices and positions must have identical indices")

    if not prices.columns.equals(positions.columns):
        raise ValueError(
            "prices and positions must have identical asset columns in the same order"
        )
    
    if not prices.index.is_unique:
        raise ValueError("The time index must be unique.")

    if not prices.columns.is_unique:
        raise ValueError("Asset column names must be unique.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("The time index must be sorted ascending.")

    #Validate configuration
    if( isinstance(initial_capital, (bool, np.bool_)) 
        or not isinstance(initial_capital, (int, float, np.integer, np.floating)) 
    ):
        raise TypeError("initial_capital must be a numeric")
    
    if not np.isfinite(initial_capital) or initial_capital <=0:
        raise ValueError("initial_capital must be finite and positive.")
    
    if(isinstance(transaction_cost_rate, (bool, np.bool_))
        or not isinstance(transaction_cost_rate, (int, float, np.integer, np.floating))
    ):
        raise TypeError("transaction cost must be numeric.")

    if(not np.isfinite(transaction_cost_rate) or not 0 <= transaction_cost_rate <1) :
        raise ValueError("transaction_cost_rate must be finite and in [0,1)")

    #individual asset returns.
    asset_returns = prices.pct_change(fill_method = None).fillna(0.0)

    #shift target weights to prevent look-ahead bias.
    held_positions = positions.shift(1).fillna(0.0)

    #gross portfolio returns.
    weighted_asset_returns = held_positions* asset_returns
    gross_returns = weighted_asset_returns.sum(axis=1)

    #Portfolio turnover
    turnover = held_positions.diff().abs().sum(axis=1)

    if not turnover.empty:
        turnover.iloc[0] = held_positions.iloc[0].abs().sum()

    #proportional transaction costs.
    transaction_costs = turnover* transaction_cost_rate

    #net profolio returns
    net_returns = gross_returns - transaction_costs

    #portfolio equity curves.
    gross_equity_curve = (initial_capital * (1.0 + gross_returns).cumprod())

    equity_curve = ( initial_capital * ( 1.0 + net_returns).cumprod())

    #Return portfolio and asset-level results.
    return {
        "asset_returns": asset_returns,
        "positions": held_positions,
        "weighted_asset_returns": weighted_asset_returns,
        "gross_returns": gross_returns,
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "net_returns": net_returns,
        "gross_equity_curve": gross_equity_curve,
        "equity_curve": equity_curve,
        "initial_capital": float(initial_capital),
    }
