"""Tests for evaluation heatmaps."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation import heatmaps


@pytest.fixture
def datetime_returns():
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.tile([0.01, -0.004, 0.006, 0.002, -0.003], 20)
    return pd.Series(values, index=index, name="returns")


def test_validate_heatmap_data_accepts_numeric_frame():
    data = pd.DataFrame([[1, 2], [3, 4]])
    result = heatmaps.validate_heatmap_data(data)
    assert result.shape == data.shape
    assert result.to_numpy(dtype=float).dtype.kind == "f"


@pytest.mark.parametrize("bad", [None, pd.Series([1, 2]), [[1, 2], [3, 4]]])
def test_validate_heatmap_data_rejects_non_dataframe(bad):
    with pytest.raises(TypeError):
        heatmaps.validate_heatmap_data(bad)


@pytest.mark.parametrize("bad", [pd.DataFrame(), pd.DataFrame([[1, np.nan]]),
                                  pd.DataFrame([[np.inf]])])
def test_validate_heatmap_data_rejects_invalid_frame(bad):
    with pytest.raises(ValueError):
        heatmaps.validate_heatmap_data(bad)


def test_validate_datetime_returns_accepts_valid_series(datetime_returns):
    result = heatmaps.validate_datetime_returns(datetime_returns)
    assert result.index.equals(datetime_returns.index)
    assert len(result) == len(datetime_returns)


def test_validate_datetime_returns_rejects_non_datetime_index():
    with pytest.raises(TypeError):
        heatmaps.validate_datetime_returns(pd.Series([0.01, 0.02], index=[1, 2]))


def test_validate_datetime_returns_rejects_duplicate_index():
    index = pd.DatetimeIndex(["2025-01-01", "2025-01-01"])
    with pytest.raises(ValueError):
        heatmaps.validate_datetime_returns(pd.Series([0.01, 0.02], index=index))


def test_validate_datetime_returns_rejects_unsorted_index():
    index = pd.DatetimeIndex(["2025-01-02", "2025-01-01"])
    with pytest.raises(ValueError):
        heatmaps.validate_datetime_returns(pd.Series([0.01, 0.02], index=index))


def test_plot_heatmap_returns_axes():
    data = pd.DataFrame([[0.1, -0.1], [0.05, 0.02]], index=[2024, 2025], columns=["A", "B"])
    ax = heatmaps.plot_heatmap(data, "Test", "Column", "Row", "Return", center_zero=True)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test"
    plt.close(ax.figure)


def test_plot_heatmap_allows_missing_when_requested():
    data = pd.DataFrame([[0.1, np.nan], [np.nan, -0.1]])
    ax = heatmaps.plot_heatmap(data, "Test", "X", "Y", "Value", allow_missing=True)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_plot_heatmap_rejects_all_missing_data():
    data = pd.DataFrame([[np.nan, np.nan]])
    with pytest.raises(ValueError):
        heatmaps.plot_heatmap(data, "Test", "X", "Y", "Value", allow_missing=True)


def test_monthly_returns_heatmap(datetime_returns):
    ax = heatmaps.plot_monthly_returns_heatmap(datetime_returns)
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_xticklabels()) == 12
    plt.close(ax.figure)


def test_day_of_week_heatmap(datetime_returns):
    ax = heatmaps.plot_day_of_week_heatmap(datetime_returns)
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_xticklabels()) == 7
    plt.close(ax.figure)


def test_hour_of_day_heatmap(datetime_returns):
    ax = heatmaps.plot_hour_of_day_heatmap(datetime_returns)
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_xticklabels()) == 24
    plt.close(ax.figure)


def test_parameter_optimization_heatmap():
    data = pd.DataFrame([[0.1, 0.2], [-0.1, 0.05]], index=[5, 10], columns=[10, 20])
    ax = heatmaps.plot_parameter_optimization_heatmap(data, metric="Sharpe")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_rolling_performance_heatmap():
    data = pd.DataFrame([[0.1, 0.2], [0.0, -0.1]], index=["Jan", "Feb"], columns=[20, 60])
    ax = heatmaps.plot_rolling_performance_heatmap(data, metric="Sharpe")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_correlation_heatmap():
    data = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03], "B": [0.02, 0.01, -0.02, 0.04]})
    ax = heatmaps.plot_correlation_heatmap(data)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_correlation_heatmap_rejects_one_asset():
    with pytest.raises(ValueError):
        heatmaps.plot_correlation_heatmap(pd.DataFrame({"A": [0.01, 0.02]}))


def test_correlation_heatmap_rejects_invalid_method():
    data = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, 0.01, -0.02]})
    with pytest.raises(ValueError):
        heatmaps.plot_correlation_heatmap(data, method="invalid")
