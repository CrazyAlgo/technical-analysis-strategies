
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import pandas as pd

from technical_analysis.indicators import (
    sma,
    ema,
    vwap,
    rsi,
    bollinger_bands,
)

df = pd.DataFrame({
    "High":   [102, 104, 103, 106, 108, 107, 110, 112, 111, 114,
               115, 113, 116, 118, 117, 120, 122, 121, 124, 126,
               125, 128, 130, 129, 132, 134, 133, 136, 138, 140],
    "Low":    [99, 101, 100, 103, 105, 104, 107, 109, 108, 111,
               112, 110, 113, 115, 114, 117, 119, 118, 121, 123,
               122, 125, 127, 126, 129, 131, 130, 133, 135, 137],
    "Close":  [101, 103, 102, 105, 107, 106, 109, 111, 110, 113,
               114, 112, 115, 117, 116, 119, 121, 120, 123, 125,
               124, 127, 129, 128, 131, 133, 132, 135, 137, 139],
    "Volume": [1000, 1200, 900, 1500, 1800, 1300, 1600, 2000, 1700, 2200,
               1800, 1400, 2100, 1900, 1600, 2300, 2500, 1800, 2400, 2600,
               2000, 2700, 2900, 2200, 3000, 3200, 2500, 3300, 3500, 3700],
})

sma_result = sma(df["Close"], window=5)
ema_result = ema(df["Close"], span=5)
vwap_result = vwap(df["High"], df["Low"], df["Close"], df["Volume"])
rsi_result = rsi(df["Close"], period=14)
bb_result = bollinger_bands(df["Close"], window=20)

print("SMA:", sma_result.tail(3).tolist())
print("EMA:", ema_result.tail(3).tolist())
print("VWAP:", vwap_result.tail(3).tolist())
print("RSI:", rsi_result.tail(3).tolist())
print("\nBollinger Bands:")
print(bb_result.tail(3).round(2))

assert sma_result.notna().any()
assert ema_result.notna().any()
assert vwap_result.notna().any()
valid_rsi = rsi_result.dropna()
assert valid_rsi.between(0, 100).all()
assert bb_result[["middle", "upper", "lower"]].notna().any().all()

print("\nAll indicator calculations passed!")
