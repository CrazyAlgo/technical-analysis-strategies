"""Integration tests for consolidated evaluation reports."""

import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation.backtest import run_backtest
from technical_analysis.evaluation.report import generate_report
from technical_analysis.evaluation import metrics, trade_analysis, transaction_costs


SERIES_KEYS = ["gross_returns", "net_returns", "turnover", "transaction_costs",
               "gross_equity_curve", "equity_curve"]


@pytest.fixture
def backtest_results():
    index = pd.date_range("2025-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAPL": [100, 102, 101, 104, 106, 105],
                           "MSFT": [200, 202, 204, 203, 207, 210]}, index=index)
    positions = pd.DataFrame({"AAPL": [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],
                              "MSFT": [0.5, 0.5, 0.0, 0.5, 0.5, 0.5]}, index=index)
    return run_backtest(prices, positions, initial_capital=100_000.0,
                        transaction_cost_rate=0.001)


@pytest.fixture
def benchmark_returns(backtest_results):
    return pd.Series([0.01, -0.005, 0.012, 0.004, -0.002, 0.006],
                      index=backtest_results["net_returns"].index)


def test_report_has_expected_sections(backtest_results):
    report = generate_report(backtest_results)
    assert set(report) == {"performance", "backtest_summary", "transaction_cost_analysis", "trade_analysis"}
    assert isinstance(report["performance"], dict)
    assert isinstance(report["backtest_summary"], dict)
    assert isinstance(report["transaction_cost_analysis"], dict)
    assert report["trade_analysis"] is None


def test_backtest_summary_matches_source(backtest_results):
    summary = generate_report(backtest_results)["backtest_summary"]
    assert summary["start_date"] == backtest_results["net_returns"].index[0]
    assert summary["end_date"] == backtest_results["net_returns"].index[-1]
    assert summary["number_of_periods"] == len(backtest_results["net_returns"])
    assert summary["initial_capital"] == pytest.approx(backtest_results["initial_capital"])
    assert summary["final_gross_equity"] == pytest.approx(backtest_results["gross_equity_curve"].iloc[-1])
    assert summary["final_net_equity"] == pytest.approx(backtest_results["equity_curve"].iloc[-1])
    assert summary["total_turnover"] == pytest.approx(backtest_results["turnover"].sum())


def test_metrics_are_forwarded_correctly(backtest_results, benchmark_returns):
    report = generate_report(backtest_results, benchmark_returns=benchmark_returns,
                             risk_free_rate=0.02, periods_per_year=252)
    expected = metrics.calculate_metrics(returns=backtest_results["net_returns"],
                                         benchmark_returns=benchmark_returns,
                                         risk_free_rate=0.02, periods_per_year=252,
                                         initial_capital=backtest_results["initial_capital"])
    assert report["performance"].keys() == expected.keys()
    for key, value in expected.items():
        assert report["performance"][key] == pytest.approx(value, nan_ok=True)


def test_cost_analysis_is_forwarded_correctly(backtest_results):
    report = generate_report(backtest_results)
    expected = transaction_costs.analyze_transaction_costs(
        transaction_costs=backtest_results["transaction_costs"],
        gross_returns=backtest_results["gross_returns"],
        net_returns=backtest_results["net_returns"])
    assert report["transaction_cost_analysis"].keys() == expected.keys()
    for key, value in expected.items():
        assert report["transaction_cost_analysis"][key] == pytest.approx(value, nan_ok=True)


def test_trade_analysis_is_forwarded_correctly(backtest_results):
    pnl = pd.Series([100.0, -50.0, 150.0, -25.0])
    holding = pd.Series([2, 3, 1, 4])
    report = generate_report(backtest_results, trade_pnl=pnl, holding_periods=holding)
    expected = trade_analysis.analyze_trades(pnl, holding)
    assert report["trade_analysis"].keys() == expected.keys()
    for key, value in expected.items():
        assert report["trade_analysis"][key] == pytest.approx(value, nan_ok=True)


@pytest.mark.parametrize("invalid", [None, [], (), "invalid", 123])
def test_non_dictionary_input_raises(invalid):
    with pytest.raises(TypeError):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS + ["initial_capital"])
def test_missing_required_key_raises(backtest_results, key):
    invalid = backtest_results.copy()
    del invalid[key]
    with pytest.raises(ValueError, match="missing required keys"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_invalid_series_type_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = [1.0, 2.0]
    with pytest.raises(TypeError, match="pandas Series"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_empty_series_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_nan_series_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = backtest_results[key].copy()
    invalid[key].iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_infinite_series_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = backtest_results[key].copy()
    invalid[key].iloc[0] = np.inf
    with pytest.raises(ValueError, match="finite values"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_nonnumeric_series_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = backtest_results[key].astype(object)
    invalid[key].iloc[0] = "not numeric"
    with pytest.raises(TypeError, match="numeric values"):
        generate_report(invalid)


@pytest.mark.parametrize("key", SERIES_KEYS)
def test_misaligned_index_raises(backtest_results, key):
    invalid = backtest_results.copy()
    invalid[key] = backtest_results[key].copy()
    invalid[key].index = pd.date_range("2026-01-01", periods=len(invalid[key]), freq="D")
    with pytest.raises(ValueError, match="same index"):
        generate_report(invalid)


def test_holding_periods_without_trade_pnl_raises(backtest_results):
    with pytest.raises(ValueError, match="trade_pnl must be supplied"):
        generate_report(backtest_results, holding_periods=pd.Series([1, 2, 3]))
