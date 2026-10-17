import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

# Allow imports from the src/ directory when testing locally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from technical_analysis.indicators import (
    sma,
    ema,
    vwap,
    rsi,
    bollinger_bands,
)

from technical_analysis.strategies import (
    sma_crossover,
    ema_crossover,
    price_vs_ma,
    vwap_crossover,
    vwap_pullback,
    rsi_strategy,
    bollinger_bands_strategy,
    gap_trading,
    breakout_3_candle,
    buy_and_hold,
)


@pytest.fixture
def sample_data():
    """Create deterministic OHLCV data for testing."""

    i = np.arange(60)
    close = 100 + 0.3 * i + 5 * np.sin(i / 2)

    open_price = close - 0.5 * np.cos(i)
    high = np.maximum(open_price, close) + 1
    low = np.minimum(open_price, close) - 1

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 100 + i,
        },
        index=pd.RangeIndex(60),
    )


def check_signals(result, original):
    """Check the basic output contract for a signal strategy."""

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(original.index)
    assert "signal" in result.columns
    assert set(result["signal"].unique()).issubset({-1, 0, 1})


# -------------------------
# Indicator tests
# -------------------------

def test_sma():
    prices = pd.Series([10, 20, 30, 40])

    result = sma(prices, window=2)

    assert result.iloc[0] != result.iloc[0]  # First value is NaN
    assert result.iloc[1] == pytest.approx(15)
    assert result.iloc[2] == pytest.approx(25)


def test_ema():
    prices = pd.Series([10, 20, 30, 40])

    result = ema(prices, span=2)

    assert result.iloc[0] == pytest.approx(10)
    assert result.iloc[1] == pytest.approx(16.6666666667)
    assert result.iloc[2] == pytest.approx(25.2631578947)


def test_vwap():
    high = pd.Series([11.0, 12.0])
    low = pd.Series([9.0, 10.0])
    close = pd.Series([10.0, 11.0])
    volume = pd.Series([100.0, 300.0])

    result = vwap(high, low, close, volume)

    # Typical prices are 10 and 11.
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(10.75)


def test_rsi():
    prices = pd.Series([10, 11, 12, 13, 14, 15], dtype=float)

    result = rsi(prices, period=3)

    assert result.iloc[-1] == pytest.approx(100.0)


def test_bollinger_bands():
    prices = pd.Series(np.arange(1, 31, dtype=float))

    result = bollinger_bands(prices, window=5, num_std=2)

    assert {"middle", "upper", "lower"}.issubset(result.columns)
    assert result["upper"].iloc[-1] > result["middle"].iloc[-1]
    assert result["middle"].iloc[-1] > result["lower"].iloc[-1]


# -------------------------
# Strategy tests
# -------------------------

def test_sma_crossover(sample_data):
    result = sma_crossover.generate_signals(
        sample_data, small_window=2, long_window=5
    )
    check_signals(result, sample_data)
    assert {"small_sma", "long_sma"}.issubset(result.columns)


def test_ema():
    prices = pd.Series([10, 20, 30, 40])
    result = ema(prices, span=2)

    # With min_periods=span, the first result is NaN.
    assert pd.isna(result.iloc[0])

    # Verify the first available EMA value.
    assert result.iloc[1] == pytest.approx(16.6666666667)


def test_price_vs_ma(sample_data):
    result = price_vs_ma.generate_signals(sample_data, window=5)
    check_signals(result, sample_data)
    assert "ma" in result.columns


def test_vwap_crossover(sample_data):
    result = vwap_crossover.generate_signals(sample_data)
    check_signals(result, sample_data)
    assert "vwap" in result.columns


def test_vwap_pullback(sample_data):
    result = vwap_pullback.generate_signals(sample_data)
    check_signals(result, sample_data)
    assert "vwap" in result.columns


def test_rsi_strategy(sample_data):
    result = rsi_strategy.generate_signals(sample_data)
    check_signals(result, sample_data)
    assert "rsi" in result.columns


def test_bollinger_bands_strategy(sample_data):
    result = bollinger_bands_strategy.generate_signals(sample_data)
    check_signals(result, sample_data)
    assert {"middle", "upper", "lower"}.issubset(result.columns)


def test_gap_trading():
    data = pd.DataFrame(
        {
            "High": [10, 11, 15, 16],
            "Low": [8, 9, 13, 6],
        }
    )

    result = gap_trading.generate_signals(data)

    # Row 2: gap up, because its low (13) > previous high (11).
    assert result.loc[2, "signal"] == 1

    # Row 3: gap down, because its high (16) is NOT below
    # the previous low (13). This is not a gap down.
    assert result.loc[3, "signal"] == 0


def test_three_candle_breakout():
    data = pd.DataFrame(
        {
            "High": [10, 11, 12, 13, 14],
            "Low": [8, 9, 10, 11, 12],
            "Close": [9, 10, 11, 14, 15],
        }
    )

    result = breakout_3_candle.generate_signals(data)

    check_signals(result, data)

    # Previous three highs at row 3 are 10, 11, 12.
    assert result.loc[3, "three_candle_high"] == 12
    assert result.loc[3, "signal"] == 1

    assert result.loc[4, "signal"] == 1


def test_buy_and_hold(sample_data):
    result = buy_and_hold.calculate_returns(sample_data)

    assert "log_returns" in result.columns
    assert "cumulative_log_returns" in result.columns
    assert "cumulative_returns" in result.columns

    # Initial cumulative return should be zero.
    assert result["cumulative_returns"].iloc[0] == pytest.approx(0)

    # Final cumulative return should match final price / initial price - 1.
    expected = (
        sample_data["Close"].iloc[-1]
        / sample_data["Close"].iloc[0]
        - 1
    )

    assert result["cumulative_returns"].iloc[-1] == pytest.approx(expected)


# -------------------------
# Validation tests
# -------------------------

def test_sma_rejects_invalid_window(sample_data):
    with pytest.raises(ValueError):
        sma_crossover.generate_signals(
            sample_data, small_window=5, long_window=5
        )


def test_strategy_rejects_missing_close():
    bad_data = pd.DataFrame({"Open": [10, 11]})

    with pytest.raises(ValueError):
        price_vs_ma.generate_signals(bad_data, window=2)
