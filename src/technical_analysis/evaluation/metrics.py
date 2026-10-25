"Performance metrics for trading strategies"

import numpy as np
import pandas as pd

def validate_returns(returns: pd.Series) -> pd.Series:
    """Validate periodic portfolio returns."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas series.")

    if returns.empty:
        raise ValueError("returns cannot be empty")

    if returns.isna().any():
        raise ValueError("returns cannot contain missing value(s).")
    
    values = returns.to_numpy(dtype = float)

    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")

    if (values < -1).any():
        raise ValueError("returns cannot be below -100%.")

    return pd.Series(values, index = returns.index)


def validate_periods_per_year(periods_per_year: int) -> int:
    """Validate the number of return periods per year."""

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, (int, np.integer))
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive integer.")

    return int(periods_per_year)

# Returns & Profitability

def cumulative_return(returns:pd.Series) -> float:
    """Calculate the cumulative compounded return."""

    returns = validate_returns(returns)
    return float((1+returns).prod()-1 )


def cagr(returns:pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the compound annual growth rate."""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year) 

    growth_factor = float((1+returns).prod())

    if growth_factor == 0:
        return -1.0

    years = len(returns)/periods_per_year

    return float ( growth_factor** (1/years) -1)


def annualized_arithmetic_return(returns: pd.Series, periods_per_year: int = 252) -> float: 
    """Calculate the annualized arithmetic mean return."""

    periods_per_year = validate_periods_per_year(periods_per_year)   
    returns = validate_returns(returns)

    return float(returns.mean() * periods_per_year)


def average_daily_return(returns: pd.Series) -> float: 
    """Calculate the average return per observation."""

    returns = validate_returns(returns)

    return float(returns.mean())


def best_period_return(returns: pd.Series) -> float: 
    """Return the highest periodic return."""

    returns = validate_returns(returns)
    
    return float(returns.max())


def worst_period_return(returns: pd.Series) -> float: 
    """Return the lowest periodic return."""

    returns = validate_returns(returns)
        
    return float(returns.min())


def positive_period_percentage(returns: pd.Series) -> float: 
    """Calculate the percentage of periods with positive returns."""

    returns = validate_returns(returns)

    return float((returns>0).mean()*100)


def total_profit_loss(returns: pd.Series, initial_capital: float) -> float: 
    """Calculate total profit or loss in currency units."""

    returns = validate_returns(returns)

    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, (int, float, np.integer, np.floating))
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError("initial_capital must be a positive finite number.")

    final_capital = initial_capital * (1+returns).prod()

    return float(final_capital - initial_capital)


# Risk Metrics

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float: 
    """Calculate annualized volatility of portfolio returns."""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    return float(returns.std(ddof=1)* np.sqrt(periods_per_year))


def maximum_drawdown(returns: pd.Series) -> float: 
    """Calculate the maximum peak-to-trough portfolio drawdown."""

    returns =  validate_returns(returns)

    wealth = (1+returns).cumprod()
    wealth = pd.concat(
        [pd.Series([1.0]), wealth.reset_index(drop = True)],
        ignore_index= True
    )

    peak = wealth.cummax()
    drawdown = wealth/peak -1

    return float(drawdown.min())


def average_drawdown(returns: pd.Series) -> float: 
    """Calculate the average drawdown across all observed periods."""

    returns = validate_returns(returns)

    wealth = (1+returns).cumprod()
    wealth = pd.concat(
        [pd.Series([1.0]), wealth.reset_index(drop = True)],
        ignore_index= True
    )

    peak = wealth.cummax()
    drawdown = wealth/peak -1

    return float(drawdown.iloc[1:].mean())


def drawdown_duration(returns: pd.Series) -> int: 
    """Calculate the longest drawdown duration in periods."""

    returns = validate_returns(returns)

    wealth = (1+returns).cumprod()
    wealth = pd.concat(
        [pd.Series([1.0]), wealth.reset_index(drop = True)],
        ignore_index= True
    )

    peak = wealth.cummax()
    underwater = wealth < peak

    longest_duration = 0
    current_duration = 0

    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0

    return longest_duration


def downside_deviation(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252 ) -> float:
    """Calculate annualized downside deviation below a target return."""

    returns = validate_returns(returns)

    if (isinstance(target_return, bool)
        or not isinstance( target_return, (int, float, np.integer, np.floating) )
        or not np.isfinite(target_return)
    ):
        raise ValueError("target_return must be a finite number.")

    periods_per_year = validate_periods_per_year(periods_per_year)

    shortfalls = np.minimum(returns.to_numpy() - target_return, 0.0)
    deviation = np.sqrt(np.mean(shortfalls ** 2))

    return float(deviation * np.sqrt(periods_per_year))


def value_at_risk(returns: pd.Series, confidence_level:float = 0.95) -> float:
    """Calculate historical Value at Risk as a return threshold."""

    returns = validate_returns(returns)

    if (isinstance(confidence_level, bool)
        or not isinstance( confidence_level, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError( "confidence_level must be a finite number between 0 and 1.")
    
    return float(returns.quantile(1- confidence_level))


def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float: 
    """Calculate the mean return in the historical VaR tail."""

    returns = validate_returns(returns)

    if (isinstance(confidence_level, bool)
        or not isinstance( confidence_level, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError( "confidence_level must be a finite number between 0 and 1.")

    var_threshold = returns.quantile(1- confidence_level)
    tail_returns = returns[returns <= var_threshold]

    return float(tail_returns.mean())


def expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate expected shortfall in the historical loss tail."""

    return conditional_value_at_risk(returns, confidence_level)


def skewness(returns: pd.Series) -> float: 
    """Calculate the skewness of periodic returns."""

    returns = validate_returns(returns)

    return float(returns.skew())


def excess_kurtosis(returns: pd.Series) -> float: 
    """Calculate the excess kurtosis of periodic returns."""

    returns = validate_returns(returns)

    return float(returns.kurt())


# Risk-Adjusted Performance

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float: 
    """Calculate the annualized Sharpe ratio."""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if (isinstance(risk_free_rate, bool)
        or not isinstance( risk_free_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError("risk_free_rate must be finite and greater than -1.")

    periodic_rf = (1+ risk_free_rate) **(1/periods_per_year) -1
    excess_returns = returns - periodic_rf
    volatility = excess_returns.std(ddof=1)

    if volatility == 0 or not np.isfinite(volatility) :
        return float("nan")

    return float(excess_returns.mean()/ volatility*np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float: 
    """ Calculate the annualized Sortino ratio"""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if (isinstance(risk_free_rate, bool)
        or not isinstance( risk_free_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError("risk_free_rate must be finite and greater than -1.")

    periodic_rf = (1+ risk_free_rate) ** (1/periods_per_year) -1
    excess_returns = returns - periodic_rf
    downside = np.minimum(excess_returns.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))

    if downside_deviation == 0:
        return float("nan")

    return float(excess_returns.mean()/downside_deviation * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float: 
    """Calculate CAGR divided by the absolute maximum drawdown."""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    annual_growth = cagr(returns, periods_per_year)
    max_dd = abs(maximum_drawdown(returns))

    if max_dd == 0:
        return float("nan")
    
    return float(annual_growth/max_dd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float: 
    """ Calculate the Omega ratio relative to a return threshold."""

    returns = validate_returns(returns)

    if (isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
    ):
        raise ValueError("threshold must be a finite number.")

    gains = np.maximum(returns.to_numpy() - threshold, 0.0).sum()
    losses = np.maximum(threshold - returns.to_numpy(), 0.0).sum()

    if losses == 0:
        return float("inf") if gains > 0 else float("nan")

    return float(gains/losses)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the annualized information ratio."""

    returns = validate_returns(returns)
    benchmark_returns = validate_returns(benchmark_returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if not returns.index.equals(benchmark_returns.index):
        raise ValueError("returns and benchmark_returns must have identical indices.")

    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std(ddof =1)

    if tracking_error == 0 or not np.isfinite(tracking_error):
        return float("nan")

    return float( active_returns.mean()/ tracking_error *np.sqrt(periods_per_year) )


def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float: 
    """Calculate the Treynor ratio using estimated market beta."""

    returns = validate_returns(returns)
    benchmark_returns = validate_returns(benchmark_returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if (isinstance(risk_free_rate, bool)
        or not isinstance( risk_free_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError("risk_free_rate must be finite and greater than -1.")

    if not returns.index.equals(benchmark_returns.index):
        raise ValueError("returns and benchmark_returns must have same indices.")

    benchmark_rf = ( (( 1+ risk_free_rate) ** (1 / periods_per_year)) - 1 )

    market_excess = benchmark_returns - benchmark_rf
    strategy_excess = returns - benchmark_rf

    market_variance = market_excess.var(ddof =1)

    if market_variance == 0 or not np.isfinite(market_variance):
        return float("nan")
    
    beta = strategy_excess.cov(market_excess) / market_variance

    if beta == 0 or not np.isfinite(beta):
        return float("nan")
    
    annual_excess_return = strategy_excess.mean() * periods_per_year

    return float(annual_excess_return / beta)


# Trading activity and implementation costs

def validate_positions(positions: pd.DataFrame) -> None:
    """Validate multi-asset portfolio positions."""

    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame.")

    if positions.empty or positions.shape[1] == 0:
        raise ValueError("positions cannot be empty.")

    if positions.isna().any().any():
        raise ValueError("positions cannot contain NaN values.")

    if not np.isfinite(positions.to_numpy(dtype=float)).all():
        raise ValueError("positions must contain only finite values.")


def turnover(positions: pd.DataFrame) -> float:
    """Calculate cumulative one-way turnover."""

    validate_positions(positions)

    changes = positions.diff().abs()
    changes.iloc[0] = positions.iloc[0].abs()

    return float(changes.to_numpy().sum() / 2)


def number_of_position_changes(positions: pd.DataFrame) -> int:
    """Count timestamps at which any asset position changes."""

    validate_positions(positions)

    changes = positions.diff()
    changes.iloc[0] = positions.iloc[0]

    return int(changes.ne(0).any(axis=1).sum())


def exposure(positions: pd.DataFrame) -> pd.Series:
    """Calculate gross exposure at each timestamp."""

    validate_positions(positions)

    return positions.abs().sum(axis=1).rename("exposure")


def average_exposure(positions: pd.DataFrame) -> float:
    """Calculate average gross exposure over time."""

    return float(exposure(positions).mean())


def gross_return(gross_returns: pd.Series) -> float:
    """Calculate cumulative return before transaction costs."""

    if not isinstance(gross_returns, pd.Series):
        raise TypeError("gross_returns must be a pandas Series.")

    if gross_returns.empty:
        raise ValueError("gross_returns cannot be empty.")

    values = gross_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("gross_returns must contain finite values.")

    if (values < -1).any():
        raise ValueError("returns cannot be below -100%.")

    return float((1 + values).prod() - 1)


def net_return(net_returns: pd.Series) -> float:
    """Calculate cumulative return after transaction costs."""

    if not isinstance(net_returns, pd.Series):
        raise TypeError("net_returns must be a pandas Series.")

    if net_returns.empty:
        raise ValueError("net_returns cannot be empty.")

    values = net_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("net_returns must contain finite values.")

    if (values < -1).any():
        raise ValueError("returns cannot be below -100%.")

    return float((1 + values).prod() - 1)


def average_position_size(positions: pd.DataFrame) -> float:
    """Calculate mean absolute position weight across assets and timestamps."""

    validate_positions(positions)

    return float(positions.abs().to_numpy().mean())


# Benchmark-Relative Metrics

def validate_benchmark_returns(returns: pd.Series, benchmark_returns: pd.Series)-> tuple[pd.Series, pd.Series]:
    """Validate and align strategy and benchmark returns."""

    returns = validate_returns(returns)
    benchmark_returns = validate_returns(benchmark_returns)

    if not returns.index.equals(benchmark_returns.index):
        raise ValueError( "returns and benchmark_returns must have identical indices.")

    return returns, benchmark_returns

    
def benchmark_cumulative_return(benchmark_returns: pd.Series) -> float: 
    """Calculate the benchmark's cumulative compounded return"""

    benchmark_returns = validate_returns(benchmark_returns)

    return float((1+ benchmark_returns).prod() -1)


def benchmark_cagr(benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate the benchmark's compound annual growth rate"""

    benchmark_returns = validate_returns(benchmark_returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    growth_factor = float((1+  benchmark_returns).prod())

    if growth_factor == 0:
        return -1.0

    year = len(benchmark_returns)/periods_per_year

    return float(growth_factor ** (1 / year) -1)


def active_return(returns: pd.Series, benchmark_returns: pd.Series) -> float: 
    """Calculate cumulative return relative to the benchmark"""

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)

    return float((1+returns).prod()- ( 1 + benchmark_returns).prod())


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized tracking error.""" 

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    active_returns = returns - benchmark_returns

    return float(active_returns.std(ddof =1)* np.sqrt(periods_per_year))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float: 
    """Calculate portfolio beta relative to the benchmark."""

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)

    benchmark_variance = benchmark_returns.var(ddof = 1)

    if( benchmark_variance == 0 or not np.isfinite(benchmark_variance)):
        return float("nan")
    
    return float(returns.cov(benchmark_returns)/benchmark_variance)


def jensens_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Jensen's alpha."""

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if (isinstance(risk_free_rate, bool)
        or not isinstance( risk_free_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError( "risk_free_rate must be finite and greater than -1.")
    
    periodic_rf = ( (1 + risk_free_rate) ** (1 / periods_per_year) - 1 )

    strategy_excess = returns - periodic_rf
    benchmark_excess = benchmark_returns - periodic_rf

    benchmark_variance = benchmark_excess.var(ddof=1)

    if ( benchmark_variance == 0 or not np.isfinite(benchmark_variance) ):
        return float("nan")

    market_beta = ( strategy_excess.cov(benchmark_excess) / benchmark_variance )

    alpha_per_period = ( strategy_excess.mean() - market_beta * benchmark_excess.mean() )

    return float(alpha_per_period * periods_per_year)


def correlation_with_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """ Calculate Pearson correlation with the benchmark""" 

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)

    correlation = returns.corr(benchmark_returns)

    return float(correlation)


def up_market_capture_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate the strategy's return relative to the benchmark
    during periods when the benchmark return is positive."""

    returns, benchmark_returns = validate_benchmark_returns( returns, benchmark_returns )

    up_market = benchmark_returns > 0

    if not up_market.any():
        return float("nan")

    strategy_return = float((1+returns[up_market]).prod()-1)
    benchmark_return = float((1+benchmark_returns[up_market]).prod()-1)

    if benchmark_return == 0:
        return float("nan")

    return float(strategy_return / benchmark_return)


def down_market_capture_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float: 
    """Calculate the strategy's return relative to the benchmark
    during periods when the benchmark return is negative. """

    returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)

    down_market = benchmark_returns < 0

    if not down_market.any():
        return float("nan")

    strategy_return = float( (1 + returns[down_market]).prod() - 1 )
    benchmark_return = float( (1 + benchmark_returns[down_market]).prod() - 1 )

    if benchmark_return == 0:
        return float("nan")

    return float(strategy_return / benchmark_return)


# Master Function

def calculate_metrics(
                    returns: pd.Series,
                    benchmark_returns: pd.Series | None = None,
                    risk_free_rate: float = 0.0,
                    periods_per_year: int = 252,
                    initial_capital: float | None = None,
                ) -> dict[str, float]:
    """Calculate all applicable performance metrics."""

    returns = validate_returns(returns)
    periods_per_year = validate_periods_per_year(periods_per_year)

    if ( isinstance(risk_free_rate, bool) 
        or not isinstance( risk_free_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError( "risk_free_rate must be finite and greater than -1.")
    
    metrics = {
        # Returns and profitability
        "cumulative_return": cumulative_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annualized_arithmetic_return": annualized_arithmetic_return( returns, periods_per_year ),
        "average_period_return": average_daily_return(returns),
        "best_period_return": best_period_return(returns),
        "worst_period_return": worst_period_return(returns),
        "positive_period_percentage": positive_period_percentage(returns),

        # Risk
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "maximum_drawdown": maximum_drawdown(returns),
        "average_drawdown": average_drawdown(returns),
        "drawdown_duration": float(drawdown_duration(returns)),
        "downside_deviation": downside_deviation( returns, periods_per_year=periods_per_year),
        "value_at_risk": value_at_risk(returns),
        "conditional_value_at_risk": conditional_value_at_risk(returns),
        "skewness": skewness(returns),
        "excess_kurtosis": excess_kurtosis(returns),

        # Risk-adjusted performance
        "sharpe_ratio": sharpe_ratio( returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio( returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "omega_ratio": omega_ratio(returns),
    }

    if initial_capital is not None:
        metrics["total_profit_loss"] = total_profit_loss( returns, initial_capital )
        
    if benchmark_returns is not None:
        returns, benchmark_returns = validate_benchmark_returns(returns, benchmark_returns)

        metrics.update({
            # Benchmark-relative performance
            "benchmark_cumulative_return": benchmark_cumulative_return( benchmark_returns ),
            "benchmark_cagr": benchmark_cagr( benchmark_returns, periods_per_year ),
            "active_return": active_return( returns, benchmark_returns ),
            "tracking_error": tracking_error( returns, benchmark_returns, periods_per_year ),
            "beta": beta(returns, benchmark_returns),
            "jensens_alpha": jensens_alpha( returns, benchmark_returns, risk_free_rate, periods_per_year ),
            "correlation_with_benchmark": correlation_with_benchmark( returns, benchmark_returns ),
            "up_market_capture_ratio": up_market_capture_ratio( returns, benchmark_returns ),
            "down_market_capture_ratio": down_market_capture_ratio( returns, benchmark_returns ),
            "information_ratio": information_ratio( returns, benchmark_returns, periods_per_year ),
        })

    return metrics