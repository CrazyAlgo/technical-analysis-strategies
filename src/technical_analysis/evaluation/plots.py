"""Plotting utilities for evaluating algorithmic trading strategies."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from technical_analysis.utils.plotting_utils import new_axes
from technical_analysis.utils.validation import validate_series, validate_positive_integer


def validate_matching_indexes(first: pd.Series, second: pd.Series, first_name : str, second_name : str)-> None:
    """ Require two Series to have identiacal indexes."""

    if not first.index.equals(second.index):
        raise ValueError(f"{first_name} series & {second_name} series must have matching indexes.")


#1. Equity curve (linear or logarithmic scale)

def plot_equity_curve( equity_curve: pd.Series, scale: str ="linear", ax = None) -> plt.Axes:
    """Plot portfolio equity on a linear or logarithmic scale."""
    
    equity_curve = validate_series(equity_curve)
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be either 'linear' or 'log'.")

    if ( equity_curve < 0 ).any():
        raise ValueError("equity_curve must not contain negative values.")

    if scale == "log" and (equity_curve <= 0).any():
        raise ValueError("A logarithmic equity plot requires striclty positive values.")

    ax = new_axes(ax)

    ax.plot(equity_curve.index, equity_curve, label="Portfolio Equity")
    ax.set_yscale(scale)
    ax.set_title(f"Portfolio Equity Curve ({scale.capitalize()} Scale)")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha = 0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax 


#2. Drawdown / underwater plot

def plot_drawdown(equity_curve: pd.Series, ax = None) -> plt.Axes:
    """Plot percetage drawdown from the previous equity peak."""

    equity_curve = validate_series(equity_curve)

    if (equity_curve <= 0). any():
        raise ValueError("equity_curve must contain positive values.")

    running_peak = equity_curve.cummax()
    drawdown = equity_curve/ running_peak -1

    ax = new_axes(ax, figsize = (12, 5))

    ax.plot(drawdown.index, drawdown, label = "Drawdown")
    ax.fill_between( drawdown.index, drawdown.to_numpy(), 0, alpha = 0.2 )
    ax.set_title("Portfolio Drawdown (Underwater Plot)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#3. Rolling Sharpe ratio

def plot_rolling_sharpe( returns: pd.Series, window: int = 63, periods_per_year: int = 252, risk_free_rate: float = 0.0, ax=None ) -> plt.Axes:
    """Plot rolling annualized Sharpe ratio.
        risk_free_rate is an annualized rate expressed as a decimal.
    """

    returns =  validate_series(returns, returns = True)
    validate_positive_integer(window, "window")
    validate_positive_integer(periods_per_year, "periods_per_year")

    if not np.isscalar(risk_free_rate) or isinstance(risk_free_rate, (bool, np.bool_)):
        raise TypeError("risk_free_rate must be a numeric scalar.") 

    risk_free_rate = float(risk_free_rate)

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1.")

    periodic_rf = (1 + risk_free_rate) ** (1 / periods_per_year) -1

    excess_returns = returns - periodic_rf
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = excess_returns.rolling(window).std(ddof=1)

    rolling_sharpe = ( rolling_mean / rolling_std.replace(0, np.nan))*np.sqrt(periods_per_year)

    ax = new_axes(ax, figsize = (12,5))

    ax.plot(rolling_sharpe.index, rolling_sharpe, label = "Rolling Sharpe")
    ax.axhline(0, linewidth =1)
    ax.set_title(f"Rolling Sharpe Ratio ({window}- Period window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#4. Rolling volatility

def plot_rolling_volatility(returns: pd.Series, window: int = 21, periods_per_year: int = 252, ax = None) -> plt.Axes:
    """Plot rolling annualized volatitlity."""

    returns = validate_series(returns, returns = True)
    validate_positive_integer(window, "window")
    validate_positive_integer(periods_per_year, "periods_per_year")

    rolling_volatility = ( returns.rolling(window).std(ddof = 1)* np.sqrt(periods_per_year))
    ax = new_axes(ax, figsize=(12, 5))

    ax.plot(rolling_volatility.index, rolling_volatility, label= "Rolling Volatility")
    ax.set_title(f"Rolling Annualized Volatility ({window}-Period Window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#5. Rolling beta against a benchmark

def plot_rolling_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63, ax = None) -> plt.Axes:
    """Plot rolling beta of strategy returns against benchmark returns."""

    strategy_returns = validate_series(strategy_returns, returns = True)
    benchmark_returns = validate_series(benchmark_returns, returns = True)
    validate_matching_indexes(strategy_returns, benchmark_returns, "strategy_returns", "benchmark_returns")
    validate_positive_integer(window, "window")

    covariance = strategy_returns.rolling(window).cov(benchmark_returns)
    benchmark_variance = benchmark_returns.rolling(window).var(ddof = 1)

    rolling_beta = covariance/ benchmark_variance.replace(0, np.nan)

    ax = new_axes(ax, figsize = (12,5))

    ax.plot(rolling_beta.index, rolling_beta, label="Rolling Beta")
    ax.axhline(1, linestyle="--", linewidth=1, label="Beta = 1")
    ax.set_title(f"Rolling Beta ({window}-Period Window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#6. Return histogram / optional KDE

def plot_returns_distribution(returns: pd.Series, bins: int = 50, kde: bool = False, ax = None) -> plt.Axes:
    """Plot periodic returns as a histogram, optionally with a KDE."""

    returns = validate_series(returns, returns = True)
    validate_positive_integer(bins, "bins")

    if kde:
        if returns.nunique() < 2:
            raise ValueError("KDE requires at least two distinct return values.")

        try:
            from scipy.stats import gaussian_kde
        except ImportError as exc:
            raise ImportError(
                "Install scipy to use KDE: pip install scipy"
            ) from exc
        
    ax = new_axes(ax, figsize=(12,5))

    ax.hist(returns.to_numpy(), bins = bins, density = kde, alpha = 0.65, label = "Returns")

    if kde:
        x_values = np.linspace(returns.min(), returns.max(), 300)
        density = gaussian_kde(returns.to_numpy())

        ax.plot(x_values, density(x_values), label="KDE")

    ax.axvline(0, linestyle="--", linewidth=1, label="Zero Return")
    ax.set_title("Distribution of Periodic Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density" if kde else "Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#7.Normal Q-Q plot

def plot_qq(returns: pd.Series, ax = None)-> plt.Axes:
    """Compare empirical return quantitles with theoretical normal quantiles"""

    
    returns = validate_series(returns, returns = True)

    if len(returns) < 2:
            raise ValueError("At least two observations are required for a Q-Q plot.")

    
    try: 
        from scipy import stats
    except ImportError as exc:
        raise ImportError("INstall scipy to create a Q-Q plot: pip install scipy.") from exc

    theoretical, ordered = stats.probplot(returns.to_numpy(), dist= "norm", fit = False)
    ax = new_axes(ax)

    ax.scatter(theoretical, ordered, s=18, alpha=0.7)

    #Reference line based on the sample mean and standard deviation.
    x_line = np.asarray([min(theoretical), max(theoretical)])
    y_line = returns.mean() + returns.std(ddof =1)*x_line

    ax.plot(x_line, y_line, linestyle= "--", label = "Normal reference")
    ax.set_title("Normal Q-Q Plot of Returns")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Observed Return Quantiles")
    ax.grid(True, alpha = 0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#8. Autocorrelation function (ACF)

def plot_acf(returns: pd.Series, lags: int = 40, ax= None) -> plt.Axes:
    """Plot sample autocorrelation of periodic returns.
    Dsshed bounds are approximate 95% white-noise bounds.
    """

    returns = validate_series(returns, returns = True)
    validate_positive_integer(lags, "lags")

    values = returns.to_numpy(dtype = float)

    if len(values) < 3:
        raise ValueError("At least three abservations are required.")

    if np.std(values) == 0:
        raise ValueError("ACF is undefined for constant returns.")

    lags = min(lags, len(values) -1)
    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    acf_values = np.array([ np.dot(centered[lag:], centered[:len(centered) - lag]) /denominator 
                           for lag in range(lags+1) ])

    ax = new_axes(ax, figsize=(10, 5))

    ax.bar(np.arange(lags + 1), acf_values, width=0.7)
    confidence = 1.96 / np.sqrt(len(values))
    ax.axhline(confidence, linestyle="--", linewidth=1)
    ax.axhline(-confidence, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_title("Autocorrelation Function of Returns")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.figure.tight_layout()

    return ax


#9.Trade P&L histogram

def plot_trade_pnl_distribution( trade_pnl: pd.Series, bins: int = 30, ax = None) -> plt.Axes:
    """Plot the distribution of realized P&L across completed trades."""

    trade_pnl = validate_series(trade_pnl)
    validate_positive_integer(bins, "bins")

    ax = new_axes(ax, figsize = (10,5))

    ax.hist(trade_pnl.to_numpy(), bins=bins, alpha=0.7)
    ax.axvline(0, linestyle="--", linewidth=1, label="Breakeven")
    ax.set_title("Trade P&L Distribution")
    ax.set_xlabel("Realized Trade P&L")
    ax.set_ylabel("Number of Trades")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#10.Gross vs. net equity curves

def plot_gross_vs_net_equity( gross_returns: pd.Series, net_returns: pd.Series, initial_capital: float = 1.0, ax = None) -> plt.Axes:
    """Compare equity curves compounded from gross and net returns."""

    gross_returns = validate_series(gross_returns, returns = True)
    net_returns = validate_series(net_returns, returns = True)
    validate_matching_indexes(gross_returns, net_returns, "gross_returns", "net_returns")

    if not np.isscalar(initial_capital) or isinstance(initial_capital, (bool, np.bool_)):
        raise TypeError("initial_capital must be a numeric scalar.")

    initial_capital = float(initial_capital)

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive.")

    gross_equity = initial_capital* (1+ gross_returns).cumprod()
    net_equity = initial_capital* (1+ net_returns).cumprod()

    ax = new_axes(ax)

    ax.plot(gross_equity.index, gross_equity, label="Gross Equity")
    ax.plot(net_equity.index, net_equity, label="Net Equity")
    ax.set_title("Gross vs. Net Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#11. In-sample vs. out-of-sample equity curve

def plot_is_oos_equity( in_sample_returns: pd.Series, out_of_sample_returns: pd.Series, initial_capital: float = 1.0, ax = None) -> plt.Axes:
    """Plot in-sample and out-of-sample equity as a continuous curve.
    The out-of-sample segment starts at the final in-sample equity value.
    """

    in_sample_returns = validate_series(in_sample_returns, returns = True)
    out_of_sample_returns = validate_series(out_of_sample_returns, returns = True)

    if not isinstance(in_sample_returns.index, pd.DatetimeIndex):
        raise TypeError("in_Sample_returns must have a DatetimeIndex.")

    if not isinstance(out_of_sample_returns.index, pd.DatetimeIndex):
        raise TypeError("out_of_sample_returns must have a DatetimeIndex")

    if not in_sample_returns.index.is_monotonic_increasing:
        raise ValueError("In-sample index must be chronological.")

    if not out_of_sample_returns.index.is_monotonic_increasing:
        raise ValueError("Out-of-sample index must be chronological.")

    if in_sample_returns.index.max() >= out_of_sample_returns.index.min():
        raise ValueError("Out-of-sample dates must begin after in-sample dates.")

    if not np.isscalar(initial_capital) or isinstance(initial_capital, (bool, np.bool_)):
        raise TypeError("initial_capital must be a numeric scalar.")

    initial_capital = float(initial_capital)

    if not np.isfinite(initial_capital) or initial_capital <=0:
        raise ValueError("initial_capital must be finite and positive.")

    in_sample_equity = initial_capital* (1+ in_sample_returns).cumprod()

    out_of_sample_equity = in_sample_equity.iloc[-1]*(1+ out_of_sample_returns).cumprod()

    ax = new_axes(ax)

    ax.plot(in_sample_equity.index, in_sample_equity, label="In-Sample")
    ax.plot( out_of_sample_equity.index, out_of_sample_equity, label="Out-of-Sample")

    ax.axvline( out_of_sample_returns.index[0], linestyle="--", linewidth=1, label="Out-of-Sample Start" )
    ax.set_title("In-Sample vs. Out-of-Sample Equity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#12.Portfolio exposure over time

def plot_exposure( positions: pd.DataFrame, ax = None) -> plt.Axes:
    """
    Plot gross exposure, net exposure and leverage proxy.
    Positions should contain portfolio weights by asset and timestamp.
    Gross exposure is the sum of absolute weights; net exposure is the
    sum of signed weights. Gross exposure acts as a leverage proxy when
    weights are expressed relative to portfolio equity.
    """
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame.")

    if positions.empty or positions.shape[1] == 0:
        raise ValueError("positions cannot be empty.")

    if positions.isna().any().any():
        raise ValueError("positions cannot contain missing values.")

    values = positions.to_numpy(dtype = float)

    if not np.isfinite(values).all():
        raise ValueError("positions must contain only finite values.")

    numeric_positions = pd.DataFrame( values, index=positions.index, columns=positions.columns)
    
    gross_exposure = numeric_positions.abs().sum(axis =1)
    net_exposure = numeric_positions.sum(axis =1)

    ax = new_axes(ax)

    ax.plot(gross_exposure.index, gross_exposure, label="Gross Exposure")
    ax.plot(net_exposure.index, net_exposure, label="Net Exposure")
    ax.set_title("Portfolio Exposure Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Exposure (Portfolio Weight)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#13. Monte Carlo / bootstrap fan chart

def plot_bootstrap_fan(
    simulated_paths: pd.DataFrame,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    ax=None,
) -> plt.Axes:
    """ Plot percentile bands from precomputed simulated equity paths.
    Rows represent time; columns represent independent simulation paths.
    """

    if not isinstance(simulated_paths, pd.DataFrame):
        raise TypeError("simulated_paths must be a pandas DataFrame.")

    if simulated_paths.empty or simulated_paths.shape[1] == 0:
        raise ValueError("simulated_paths cannot be empty.")

    if simulated_paths.isna().any().any():
        raise ValueError("simulated_paths cannot contain missing values.")

    values = simulated_paths.to_numpy(dtype = float)

    if not np.isfinite(values).all():
        raise ValueError("simulated_paths must contain only finite values.")

    for name, value in (
        ("lower_percentile", lower_percentile),
        ("upper_percentile", upper_percentile),
    ):
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
        ):
            raise TypeError(f"{name} must be a numeric scalar.")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        
    if not 0 <= lower_percentile < upper_percentile <=100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <=100.")

    lower = np.percentile(values, lower_percentile, axis = 1)
    median = np.percentile(values, 50, axis =1)
    upper = np.percentile(values, upper_percentile, axis = 1)

    ax = new_axes(ax)

    ax.fill_between( simulated_paths.index, lower, upper, alpha=0.25, label=f"{lower_percentile:g}th–{upper_percentile:g}th Percentile")
    ax.plot(simulated_paths.index, median, label="Median Path")
    ax.set_title("Monte Carlo / Bootstrap Fan Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Simulated Portfolio Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax


#14. Maximum adverse excursion vs. maximum favorable excursion

def plot_mae_mfe(trades: pd.DataFrame, mae_column: str = "mae", mfe_column: str = "mfe", ax = None) -> plt.Axes:
    """Plot maximum adverse excursion against maximum favorable excursion.

    Both columns must use the same units, such as currency or return.
    MAE is expected to be non-positive and MFE non-negative.
    """

    if not isinstance(trades, pd.DataFrame):
        raise TypeError("trades must be a pandas DataFrame.")

    if trades.empty:
        raise ValueError("trades cannot be empty.")

    for column in (mae_column, mfe_column):
        if column not in trades.columns:
            raise ValueError(f"Missing required columns: {column}")

    if trades[[mae_column, mfe_column]].isna().any().any():
        raise ValueError("MAE and MFE values cannot contain missing values.")

    mae = trades[mae_column].to_numpy(dtype = float)
    mfe = trades[mfe_column].to_numpy(dtype = float)

    if not np.isfinite(mae).all() or not np.isfinite(mfe).all():
        raise ValueError("MAE and MFE values must be finite.")

    if (mae > 0).any():
        raise ValueError("MAE values should be non-positive.")

    if (mfe < 0).any():
        raise ValueError("MFE values should be non-negative.")

    ax = new_axes(ax, figsize = (8,6))

    ax.scatter(mae, mfe, alpha=0.65)
    ax.set_title("Maximum Adverse vs. Favorable Excursion")
    ax.set_xlabel("Maximum Adverse Excursion (MAE)")
    ax.set_ylabel("Maximum Favorable Excursion (MFE)")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()

    return ax


#15. Spread, z-score bands, and entry/exit markers

def plot_spread_zscore(
        spread: pd.Series,
        zscore: pd.Series,
        entry_signal: pd.Series | None = None,
        exit_signal: pd.Series | None = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        ax= None,
        spread_ax=None,
    ) -> plt.Axes:

    """
    Plot the raw spread and its z-score with thresholds and optional entry/exit markers.
    Signal Series should contain booleans or numeric codes:
        0/1 for entries and 0/-1 for exits, aligned with the spread and z-score indexes.
    Provide both ax and spread_ax, or neither.
    """

    spread = validate_series(spread)
    zscore = validate_series(zscore)

    validate_matching_indexes(spread, zscore, "spread", "zscore")

    for name, value in ( ("entry_threshold", entry_threshold), ("exit_threshold", exit_threshold)):
        if ( not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_))):
            raise TypeError(f"{name} must be a numeric scalar.")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        
    if entry_threshold <= 0:
        raise ValueError("entry_threshold must be positive")

    if abs(exit_threshold) >= entry_threshold:
        raise ValueError( "exit_threshold magnitude must be smaller than entry_threshold." )

    # axes
    if ax is None and spread_ax is None:
        fig, (spread_ax, ax) = plt.subplots( 2, 1, figsize=(12, 8), sharex=True )
    elif ax is None or spread_ax is None:
        raise ValueError("Provide both ax and spread_ax, or neither.")
    else:
        if ax is spread_ax:
            raise ValueError("ax and spread_ax must be different axes.")

        if ax.figure is not spread_ax.figure:
            raise ValueError("Both axes must belong to the same figure.")

        fig = ax.figure

    spread_ax.plot(spread.index, spread, label="Spread")
    spread_ax.set_title("Raw Spread")
    spread_ax.set_ylabel("Spread Value")
    spread_ax.grid(True, alpha=0.3)
    spread_ax.legend()

    ax.plot(zscore.index, zscore, label="Spread Z-Score")
    ax.axhline(entry_threshold, linestyle="--", label="Upper Entry Threshold")
    ax.axhline(-entry_threshold, linestyle="--", label="Lower Entry Threshold")
    ax.axhline(exit_threshold, linestyle=":", label="Exit Threshold")

    if exit_threshold != 0:
        ax.axhline(-exit_threshold, linestyle=":", label="Negative Exit Threshold")

    if entry_signal is not None:
        if not isinstance(entry_signal, pd.Series):
            raise TypeError("entry_signal must be a pandas series.")

        validate_matching_indexes(spread, entry_signal, "spread", "entry_signal")
        validate_matching_indexes(zscore, entry_signal, "zscore", "entry_signal")

        if pd.api.types.is_bool_dtype(entry_signal.dtype):
            entries = entry_signal.fillna(False)
        else:
            valid_values = {0, 1}
            signals = entry_signal.dropna()

            if not signals.isin(valid_values).all():
                raise ValueError("entry_signal must contain only 0 or 1.")

            entries = entry_signal.fillna(0).astype(bool)

        ax.scatter(zscore.index[entries], zscore.loc[entries], marker="^", label="Entry", zorder=3 )

    if exit_signal is not None:
        if not isinstance(exit_signal, pd.Series):
            raise TypeError("exit_signal must be a pandas Series.")

        validate_matching_indexes(spread, exit_signal, "spread", "exit_signal")
        validate_matching_indexes(zscore, exit_signal, "zscore", "exit_signal")

        if pd.api.types.is_bool_dtype(exit_signal.dtype):
            exits = exit_signal.fillna(False)
        else:
            valid_values = {0, -1}

            if not signals.isin(valid_values).all():
                raise ValueError("exit_signal must contain only 0 or -1.")

            exits = exit_signal.fillna(0).astype(bool)

        ax.scatter( zscore.index[exits], zscore.loc[exits], marker="x", label="Exit", zorder=3 )

    ax.set_title("Spread Z-Score with Entry and Exit Signals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return ax