"""Tests for performance, risk, and benchmark-relative metrics."""

import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation import metrics as m


@pytest.fixture
def returns():
    return pd.Series([0.10, -0.05, 0.04, 0.02], index=pd.RangeIndex(4))


@pytest.fixture
def benchmark_returns():
    return pd.Series([0.03, -0.02, 0.01, 0.015], index=pd.RangeIndex(4))


def test_validate_returns_accepts_valid_series(returns):
    result = m.validate_returns(returns)
    pd.testing.assert_series_equal(result, returns.astype(float), check_names=False)


@pytest.mark.parametrize("bad", [None, [0.1, 0.2], pd.DataFrame({"r": [0.1]})])
def test_validate_returns_rejects_non_series(bad):
    with pytest.raises(TypeError):
        m.validate_returns(bad)


@pytest.mark.parametrize("bad", [pd.Series(dtype=float), pd.Series([0.1, np.nan]),
                                  pd.Series([0.1, np.inf]), pd.Series([-1.01])])
def test_validate_returns_rejects_invalid_values(bad):
    with pytest.raises((ValueError, TypeError)):
        m.validate_returns(bad)


def test_cumulative_return(returns):
    assert m.cumulative_return(returns) == pytest.approx(np.prod(1 + returns) - 1)


def test_cagr(returns):
    expected = np.prod(1 + returns) ** (252 / len(returns)) - 1
    assert m.cagr(returns) == pytest.approx(expected)


def test_annualized_arithmetic_return(returns):
    assert m.annualized_arithmetic_return(returns, 252) == pytest.approx(returns.mean() * 252)


def test_average_daily_return(returns):
    assert m.average_daily_return(returns) == pytest.approx(returns.mean())


def test_best_and_worst_period_return(returns):
    assert m.best_period_return(returns) == pytest.approx(0.10)
    assert m.worst_period_return(returns) == pytest.approx(-0.05)


def test_positive_period_percentage(returns):
    assert m.positive_period_percentage(returns) == pytest.approx(75.0)


def test_total_profit_loss(returns):
    initial = 10_000
    expected = initial * np.prod(1 + returns) - initial
    assert m.total_profit_loss(returns, initial) == pytest.approx(expected)


def test_annualized_volatility(returns):
    assert m.annualized_volatility(returns, 252) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))


def test_maximum_drawdown_is_nonpositive(returns):
    assert m.maximum_drawdown(returns) <= 0
    assert m.maximum_drawdown(pd.Series([0.1, 0.1, 0.1])) == pytest.approx(0.0)


def test_average_drawdown_is_nonpositive(returns):
    assert m.average_drawdown(returns) <= 0


def test_drawdown_duration(returns):
    assert isinstance(m.drawdown_duration(returns), (int, np.integer))
    assert m.drawdown_duration(pd.Series([0.1, 0.1, 0.1])) == 0


def test_downside_deviation_nonnegative(returns):
    assert m.downside_deviation(returns) >= 0


def test_var_and_cvar_are_return_thresholds(returns):
    var = m.value_at_risk(returns, 0.95)
    cvar = m.conditional_value_at_risk(returns, 0.95)
    assert np.isfinite(var)
    assert np.isfinite(cvar)
    assert cvar <= returns.max()
    assert m.expected_shortfall(returns, 0.95) == pytest.approx(cvar)


def test_skewness_and_kurtosis_match_pandas(returns):
    assert m.skewness(returns) == pytest.approx(returns.skew())
    assert m.excess_kurtosis(returns) == pytest.approx(returns.kurt())


def test_sharpe_ratio_is_finite_for_variable_returns(returns):
    assert np.isfinite(m.sharpe_ratio(returns))


def test_sortino_ratio_is_float_or_nan(returns):
    value = m.sortino_ratio(returns)
    assert isinstance(value, (float, np.floating))


def test_calmar_ratio_is_float_or_nan(returns):
    value = m.calmar_ratio(returns)
    assert isinstance(value, (float, np.floating))


def test_omega_ratio_nonnegative(returns):
    assert m.omega_ratio(returns) >= 0


def test_information_ratio_with_aligned_benchmark(returns, benchmark_returns):
    value = m.information_ratio(returns, benchmark_returns)
    assert isinstance(value, (float, np.floating))


def test_treynor_ratio_with_aligned_benchmark(returns, benchmark_returns):
    value = m.treynor_ratio(returns, benchmark_returns)
    assert isinstance(value, (float, np.floating))


def test_position_metrics(returns):
    positions = pd.DataFrame({"A": [0.5, 0.5, 0.0, -0.5], "B": [0.0, 0.5, 0.5, 0.0]})
    assert m.turnover(positions) >= 0
    assert m.number_of_position_changes(positions) > 0
    assert len(m.exposure(positions)) == len(positions)
    assert m.average_exposure(positions) >= 0
    assert m.average_position_size(positions) >= 0


def test_gross_and_net_return_helpers(returns):
    assert m.gross_return(returns) == pytest.approx(np.prod(1 + returns) - 1)
    assert m.net_return(returns) == pytest.approx(np.prod(1 + returns) - 1)


def test_benchmark_relative_metrics(returns, benchmark_returns):
    assert m.benchmark_cumulative_return(benchmark_returns) == pytest.approx(np.prod(1 + benchmark_returns) - 1)
    assert isinstance(m.benchmark_cagr(benchmark_returns), float)
    assert isinstance(m.active_return(returns, benchmark_returns), float)
    assert m.tracking_error(returns, benchmark_returns) >= 0
    assert isinstance(m.beta(returns, benchmark_returns), float)
    assert isinstance(m.jensens_alpha(returns, benchmark_returns), float)
    assert isinstance(m.correlation_with_benchmark(returns, benchmark_returns), float)
    assert isinstance(m.up_market_capture_ratio(returns, benchmark_returns), float)
    assert isinstance(m.down_market_capture_ratio(returns, benchmark_returns), float)


def test_benchmark_index_mismatch_raises(returns, benchmark_returns):
    benchmark_returns.index = [10, 11, 12, 13]
    with pytest.raises(ValueError, match="indices"):
        m.validate_benchmark_returns(returns, benchmark_returns)


def test_calculate_metrics_without_benchmark(returns):
    result = m.calculate_metrics(returns, initial_capital=10_000)
    assert isinstance(result, dict)
    assert "cumulative_return" in result
    assert "maximum_drawdown" in result
    assert "total_profit_loss" in result
    assert "beta" not in result


def test_calculate_metrics_with_benchmark(returns, benchmark_returns):
    result = m.calculate_metrics(returns, benchmark_returns=benchmark_returns, initial_capital=10_000)
    assert "beta" in result
    assert "tracking_error" in result
    assert "benchmark_cumulative_return" in result


@pytest.mark.parametrize("periods", [0, -1, 1.5, True, "252"])
def test_invalid_periods_per_year_raise(returns, periods):
    with pytest.raises((ValueError, TypeError)):
        m.cagr(returns, periods)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan, True])
def test_invalid_var_confidence_raises(returns, confidence):
    with pytest.raises((ValueError, TypeError)):
        m.value_at_risk(returns, confidence)


@pytest.mark.parametrize("risk_free_rate", [-1, -2, np.inf, np.nan, True, "0.02"])
def test_invalid_risk_free_rate_raises(returns, risk_free_rate):
    with pytest.raises((ValueError, TypeError)):
        m.sharpe_ratio(returns, risk_free_rate=risk_free_rate)
