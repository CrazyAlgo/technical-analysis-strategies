"""Unit tests for the multi-asset backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation.backtest import run_backtest


@pytest.fixture
def market_data():
    index = pd.date_range("2025-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0, 108.9], "BBB": [50.0, 50.0, 55.0, 60.5]},
        index=index,
    )
    positions = pd.DataFrame(
        {"AAA": [1.0, 1.0, 0.0, -0.5], "BBB": [0.0, 0.0, 1.0, 0.5]},
        index=index,
    )
    return prices, positions


def test_run_backtest_returns_expected_keys(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions)
    expected = {
        "asset_returns", "positions", "weighted_asset_returns", "gross_returns",
        "turnover", "transaction_costs", "net_returns", "gross_equity_curve",
        "equity_curve", "initial_capital",
    }
    assert expected.issubset(result)


def test_run_backtest_output_shapes_and_indices(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions)
    for key in ("gross_returns", "net_returns", "turnover", "transaction_costs",
                "gross_equity_curve", "equity_curve"):
        assert isinstance(result[key], pd.Series)
        assert result[key].index.equals(prices.index)
        assert len(result[key]) == len(prices)
    assert result["asset_returns"].shape == prices.shape
    assert result["positions"].shape == positions.shape
    assert result["weighted_asset_returns"].shape == prices.shape


def test_positions_are_lagged_to_avoid_lookahead(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions)
    pd.testing.assert_series_equal(
        result["positions"].iloc[1], positions.iloc[0], check_names=False
    )


def test_returns_and_equity_are_finite(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions)
    for key in ("gross_returns", "net_returns", "turnover", "transaction_costs",
                "gross_equity_curve", "equity_curve"):
        assert np.isfinite(result[key].to_numpy(dtype=float)).all()


def test_zero_costs_make_gross_and_net_returns_equal(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions, transaction_cost_rate=0.0)
    pd.testing.assert_series_equal(result["gross_returns"], result["net_returns"])
    assert (result["transaction_costs"] == 0).all()


def test_costs_do_not_improve_period_return(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions, transaction_cost_rate=0.001)
    assert (result["net_returns"] <= result["gross_returns"] + 1e-12).all()
    assert (result["transaction_costs"] >= 0).all()


def test_initial_capital_is_preserved(market_data):
    prices, positions = market_data
    result = run_backtest(prices, positions, initial_capital=250_000.0)
    assert result["initial_capital"] == 250_000.0
    assert result["equity_curve"].iloc[0] == pytest.approx(
        250_000.0 * (1 + result["net_returns"].iloc[0])
    )


@pytest.mark.parametrize("bad_prices", [None, [], pd.Series([1, 2])])
def test_invalid_prices_type_raises(bad_prices, market_data):
    _, positions = market_data
    with pytest.raises((TypeError, ValueError)):
        run_backtest(bad_prices, positions)


@pytest.mark.parametrize("bad_positions", [None, [], pd.Series([0, 1])])
def test_invalid_positions_type_raises(bad_positions, market_data):
    prices, _ = market_data
    with pytest.raises((TypeError, ValueError)):
        run_backtest(prices, bad_positions)


def test_mismatched_columns_raise(market_data):
    prices, positions = market_data
    positions = positions.rename(columns={"AAA": "OTHER"})
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions)


def test_mismatched_indices_raise(market_data):
    prices, positions = market_data
    positions.index = positions.index + pd.Timedelta(days=1)
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions)


@pytest.mark.parametrize("bad_capital", [0, -1, np.inf, np.nan, True, "100"])
def test_invalid_initial_capital_raises(market_data, bad_capital):
    prices, positions = market_data
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions, initial_capital=bad_capital)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.0, 2.0, np.inf, np.nan, True, "0.1"])
def test_invalid_transaction_cost_rate_raises(market_data, bad_rate):
    prices, positions = market_data
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions, transaction_cost_rate=bad_rate)


def test_nonpositive_prices_raise(market_data):
    prices, positions = market_data
    prices.iloc[1, 0] = 0
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions)


def test_unsorted_index_raises(market_data):
    prices, positions = market_data
    prices = prices.iloc[::-1]
    positions = positions.iloc[::-1]
    with pytest.raises((ValueError, TypeError)):
        run_backtest(prices, positions)
