"""Smoke and validation tests for evaluation plots."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation import plots


@pytest.fixture
def series_data():
    index = pd.date_range("2025-01-01", periods=100, freq="D")
    returns = pd.Series(np.tile([0.01, -0.004, 0.006, 0.002, -0.003], 20), index=index)
    equity = 10_000 * (1 + returns).cumprod()
    return returns, equity


def test_new_axes_creates_axes():
    ax = plots.new_axes()
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_new_axes_reuses_supplied_axes():
    fig, supplied = plt.subplots()
    assert plots.new_axes(supplied) is supplied
    plt.close(fig)


def test_plot_equity_curve_returns_axes(series_data):
    _, equity = series_data
    ax = plots.plot_equity_curve(equity)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1
    plt.close(ax.figure)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_plot_equity_curve_supports_scales(series_data, scale):
    _, equity = series_data
    ax = plots.plot_equity_curve(equity, scale=scale)
    assert ax.get_yscale() == scale
    plt.close(ax.figure)


def test_plot_equity_curve_rejects_invalid_scale(series_data):
    _, equity = series_data
    with pytest.raises(ValueError):
        plots.plot_equity_curve(equity, scale="invalid")


def test_plot_equity_curve_rejects_negative_equity():
    equity = pd.Series([100.0, -1.0, 90.0])
    with pytest.raises(ValueError):
        plots.plot_equity_curve(equity)


def test_plot_equity_curve_log_rejects_zero():
    equity = pd.Series([100.0, 0.0, 90.0])
    with pytest.raises(ValueError):
        plots.plot_equity_curve(equity, scale="log")


def test_plot_drawdown_returns_axes(series_data):
    _, equity = series_data
    ax = plots.plot_drawdown(equity)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1
    plt.close(ax.figure)


def test_plot_drawdown_rejects_nonpositive_equity():
    with pytest.raises(ValueError):
        plots.plot_drawdown(pd.Series([100.0, 0.0, 80.0]))


def test_plot_rolling_sharpe_returns_axes(series_data):
    returns, _ = series_data
    ax = plots.plot_rolling_sharpe(returns, window=20)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_rolling_volatility_returns_axes(series_data):
    returns, _ = series_data
    ax = plots.plot_rolling_volatility(returns, window=20)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_returns_distribution_returns_axes(series_data):
    returns, _ = series_data
    ax = plots.plot_returns_distribution(returns)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_gross_vs_net_equity_returns_axes(series_data):
    returns, _ = series_data
    ax = plots.plot_gross_vs_net_equity( gross_returns=returns, net_returns=returns - 0.0005, initial_capital=10_000)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_spread_zscore_returns_axes():
    index = pd.date_range("2025-01-01", periods=30, freq="D")
    spread = pd.Series(np.linspace(1.0, 2.0, 30), index=index)
    zscore = pd.Series(np.sin(np.linspace(0, 5, 30)), index=index)
    ax = plots.plot_spread_zscore(spread, zscore)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)
