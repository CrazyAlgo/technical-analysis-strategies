"""Heatmap utilities for evaluating algorithmic trading strategies."""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from technical_analysis.utils.plotting_utils import new_axes

def validate_heatmap_data(data: pd.DataFrame, name: str= "data") -> pd.DataFrame:
    """Validate a non-empty numeric DataFrame suitable for a heatmap."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    if data.empty:
        raise ValueError(f"{name} cannot be empty")

    if data.isna().any().any():
        raise ValueError(f"{name} cannot contain missing values.")

    try:
        values = data.to_numpy(dtype = float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")

    return data.astype(float)


def validate_datetime_returns(returns: pd.Series) -> pd.Series:
    """Validate periodic returns with a DatetimeIndex."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex.")

    if returns.index.hasnans:
        raise ValueError("returns index cannot contain NaT.")

    if returns.index.has_duplicates:
        raise ValueError("returns index cannot contain duplicate timestamps.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted chronologically.")

    if returns.isna().any():
        raise ValueError("returns cannot contain missing values.")

    try:
        values = returns.to_numpy(dtype= float)
    except(TypeError, ValueError) as exc:
        raise TypeError("returns must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError("return must contain finite values only.")

    if (values <-1).any():
        raise ValueError("returns cannot contain returns below -100%.")

    return pd.Series(values, index= returns.index, name = returns.name)

def plot_heatmap(data: pd.DataFrame,
                 title: str,
                 xlabel : str,
                 ylabel: str,
                 colorbar_label: str,
                 ax = None,
                 cmap: str = "RdYlGn",
                 center_zero: bool = False,
                 allow_missing: bool = False,
                 ) -> plt.Axes:
    """plot a validate DataFrame as a heatmap."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data cannot be empty.")

    if allow_missing:
        # Validate numeric values while permitting NaN for unobserved cells.
        try:
            values = data.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("data must contain numeric values.") from exc

        if np.isinf(values).any():
            raise ValueError("data cannot contain infinite values.")

        if np.isnan(values).all():
            raise ValueError("data must contain at least one valid value.")

    else:
        data = validate_heatmap_data(data)
        values = data.to_numpy(dtype=float)

    image_kwargs = {"cmap": cmap, "aspect" : "auto"}

    if center_zero:
        limit = np.nanmax(np.abs(values))

        if limit == 0:
            limit = 1.0

        image_kwargs.update(vmin = -limit, vmax = limit)

    masked_values = np.ma.masked_invalid(values)

    ax = new_axes(ax)

    image = ax.imshow(masked_values, **image_kwargs)

    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation = 45, ha = "right")

    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    colorbar = ax.figure.colorbar(image, ax = ax)
    colorbar.set_label(colorbar_label)

    ax.figure.tight_layout()

    return ax


#1. Parameter optimization heatmap

def plot_parameter_optimization_heatmap(
            performance: pd.DataFrame,
            metric: str,
            ax = None,
    ) -> plt.Axes:
    """ Plot a two-parameter performance grid. 
    Row will have one parameter and column will have another.
    Each cell contains the precomputed values of the selected metric """

    performance = validate_heatmap_data(performance, "performance")

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string.")

    ax = plot_heatmap(
        performance,
        title = f"Parameter Optimiazation: {metric}",
        xlabel=str(performance.columns.name or "Parameter 2"),
        ylabel=str(performance.index.name or "Parameter 1"),
        colorbar_label=metric,
        ax=ax,
        center_zero=True,
    )

    return ax


#2. Monthly returns heatmap 

def plot_monthly_returns_heatmap( returns: pd.Series, ax = None) -> plt.Axes:
    """Plot compounded monthhly returns by year and calendar month."""    

    returns = validate_datetime_returns(returns)

    monthly_returns = ( (1+returns).groupby(returns.index.to_period("M")).prod().sub(1))

    monthly_returns.index = monthly_returns.index.to_timestamp()

    table = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()

    table = table.reindex(columns = range(1,13))
    table.columns = [ "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                    ]

    if table.isna().all(axis = None):
        raise ValueError("No monthly returns are available.")

    #missing calendar months rmain blank instead od making it zero
    ax = new_axes(ax, figsize=(13, 6)) 

    values = table.to_numpy(dtype = float)
    limit = np.nanmax(np.abs(values))

    if limit == 0:
        limit = 1.0

    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="RdYlGn", #colormap red yellow green
        aspect="auto",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_xticks(range(12))
    ax.set_xticklabels(table.columns)

    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    ax.set_title("Monthly Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("Return")

    ax.figure.tight_layout()

    return ax


#3 Day of week performance heatmap 

def plot_day_of_week_heatmap( returns: pd.Series, ax = None) -> plt.Axes:
    """Plot compounded returns by weekday and calendar year."""

    returns = validate_datetime_returns(returns)

    frame = pd.DataFrame({
        "return" : returns.to_numpy(),
        "year": returns.index.year,
        "weekday": returns.index.dayofweek,
    })

    frame["period_return"] = frame["return"]

    table = (frame.groupby(["year", "weekday"])["period_return"].apply(lambda values: (1+values).prod() -1).unstack().reindex(columns=range(7)))

    table.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    return plot_heatmap(table,  
                        title="Day-of-Week Returns by Year",
                        xlabel="Day of Week",
                        ylabel="Year",
                        colorbar_label="Compounded Return",
                        ax=ax,
                        center_zero=True,
                        allow_missing=True,
                        )


#4 Hour of day performance heatmap 

def plot_hour_of_day_heatmap( returns: pd.Series, ax = None) -> plt.Axes:
    """ Plot compounded returns by hour and calendar year."""

    returns = validate_datetime_returns(returns)

    frame = pd.DataFrame({
                        "return" : returns.to_numpy(),
                        "year": returns.index.year,
                        "hour": returns.index.hour,
                        })

    table = ( frame.groupby(["year", "hour"])["return"].apply(lambda values: (1 + values).prod() - 1).unstack().reindex(columns=range(24)))

    return plot_heatmap( table,
                        title="Hour-of-Day Returns by Year",
                        xlabel="Hour (0–23)",
                        ylabel="Year",
                        colorbar_label="Compounded Return",
                        ax=ax,
                        center_zero=True,
                        allow_missing=True,
                        )


#5. Rolling performance heatmap

def plot_rolling_performance_heatmap( performance: pd.DataFrame, ax=None, metric: str = "Rolling Performance") -> plt.Axes:
    """Plot a precomputed rolling-performace matrix.
        Rows represent evaluation periods and columns represent
        rolling windows or other comparison dimensions.
        Values must be already calculated.
    """

    performance = validate_heatmap_data(performance, "performance")

    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string.")

    return plot_heatmap( performance,
                        title=metric,
                        xlabel=str(performance.columns.name or "Window"),
                        ylabel=str(performance.index.name or "Period"),
                        colorbar_label=metric,
                        ax=ax,
                        center_zero=True,
                        )


#6. Correlation matrix heatmap

def plot_correlation_heatmap( returns: pd.DataFrame, ax = None, method: str = "pearson" ) -> plt.Axes:
    """Plot the correlation matric of asset return series."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if returns.shape[1] <2:
        raise ValueError("Atleast 2 return series are required for the correlation heatmap")

    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")

    if returns.index.has_duplicates:
        raise ValueError( "returns index cannot contain duplicate timestamps." )

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError( "method must be 'pearson', 'spearman', or 'kendall'.")

    returns = validate_heatmap_data(returns, "returns")

    if(returns <-1).any().any():
        raise ValueError("returns cannot be below -100%.")

    if returns.shape[0] < 2:
        raise ValueError("At least 2 observations are required.")

    if (returns.nunique() <2).any():
        raise ValueError("Each return series must have non-zero variance.")

    correlation = returns.corr(method = method)

    return plot_heatmap(correlation,
                        title=f"Return Correlation Matrix ({method.capitalize()})",
                        xlabel="Asset",
                        ylabel="Asset",
                        colorbar_label="Correlation",
                        ax=ax,
                        cmap="RdBu_r",
                        center_zero=False,
                        )

