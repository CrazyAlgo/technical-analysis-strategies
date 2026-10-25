"""Tests for completed-trade analysis."""

import numpy as np
import pandas as pd
import pytest

from technical_analysis.evaluation import trade_analysis as ta


@pytest.fixture
def trade_pnl():
    return pd.Series([100.0, -50.0, 150.0, -25.0, 0.0])


@pytest.fixture
def holding_periods():
    return pd.Series([2, 3, 1, 4, 2])


def test_validate_trade_pnl(trade_pnl):
    result = ta.validate_trade_pnl(trade_pnl)
    pd.testing.assert_series_equal(result, trade_pnl.astype(float), check_names=False)


@pytest.mark.parametrize("bad", [None, [1.0, -1.0], pd.DataFrame({"pnl": [1.0]})])
def test_trade_pnl_rejects_non_series(bad):
    with pytest.raises(TypeError):
        ta.validate_trade_pnl(bad)


@pytest.mark.parametrize("bad", [pd.Series(dtype=float), pd.Series([1.0, np.nan]),
                                  pd.Series([np.inf]), pd.Series([-np.inf])])
def test_trade_pnl_rejects_invalid_values(bad):
    with pytest.raises((ValueError, TypeError)):
        ta.validate_trade_pnl(bad)


def test_analyze_trades_basic_values(trade_pnl, holding_periods):
    result = ta.analyze_trades(trade_pnl, holding_periods)
    assert result["total_trades"] == 5
    assert result["win_rate"] == pytest.approx(40.0)
    assert result["loss_rate"] == pytest.approx(40.0)
    assert result["average_winning_trade"] == pytest.approx(125.0)
    assert result["average_losing_trade"] == pytest.approx(-37.5)
    assert result["largest_win"] == pytest.approx(150.0)
    assert result["largest_loss"] == pytest.approx(-50.0)
    assert result["expectancy_per_trade"] == pytest.approx(35.0)
    assert result["average_holding_period"] == pytest.approx(2.4)


def test_analyze_trades_without_holding_periods(trade_pnl):
    result = ta.analyze_trades(trade_pnl)
    assert result["total_trades"] == len(trade_pnl)
    assert "average_holding_period" not in result


def test_profit_factor(trade_pnl):
    result = ta.analyze_trades(trade_pnl)
    assert result["profit_factor"] == pytest.approx(250.0 / 75.0)


def test_consecutive_win_and_loss_streaks(trade_pnl):
    result = ta.analyze_trades(trade_pnl)
    assert result["maximum_consecutive_wins"] == 1
    assert result["maximum_consecutive_losses"] == 1


def test_holding_periods_without_trade_pnl_are_not_applicable():
    with pytest.raises((TypeError, ValueError)):
        ta.analyze_trades(pd.Series([1.0]), holding_periods=pd.Series([1, 2]))


def test_holding_period_index_mismatch_raises(trade_pnl, holding_periods):
    holding_periods.index = range(10, 15)
    with pytest.raises(ValueError):
        ta.analyze_trades(trade_pnl, holding_periods)


def test_holding_periods_reject_invalid_values(trade_pnl):
    with pytest.raises((ValueError, TypeError)):
        ta.analyze_trades(trade_pnl, pd.Series([1, 2, -1, 4, 5]))
    with pytest.raises((ValueError, TypeError)):
        ta.analyze_trades(trade_pnl, pd.Series([1, 2, np.nan, 4, 5]))
