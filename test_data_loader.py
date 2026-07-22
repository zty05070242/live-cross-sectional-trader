"""
test_data_loader.py
-------------------
Pulls 30 days of 15-minute SPY bars from Alpaca and verifies the DataFrame
matches the shape that existing strategies expect via set_data().

Run with:
    python test_data_loader.py
"""

import pandas as pd
from alpaca_data_loader import load_bars
from strategies.cross_sectional import CrossSectionalStrategy

SYMBOL   = "SPY"
SYMBOL_Y = "QQQ"
DAYS     = 30
TIMEFRAME = "15Min"

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

def check_format(df: pd.DataFrame) -> list[str]:
    """Return a list of failure messages; empty list means all checks passed."""
    failures = []

    if not isinstance(df.index, pd.DatetimeIndex):
        failures.append(f"Index is {type(df.index).__name__}, expected DatetimeIndex")

    if df.index.tz is None:
        failures.append("DatetimeIndex is timezone-naive, expected timezone-aware")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        failures.append(f"Missing columns: {missing}")

    if df.empty:
        failures.append("DataFrame is empty")

    if df[list(REQUIRED_COLUMNS & set(df.columns))].isnull().any().any():
        failures.append("DataFrame contains NaN values in OHLCV columns")

    return failures


def main():
    print(f"Fetching {DAYS} days of {TIMEFRAME} bars for {SYMBOL}...")
    df = load_bars(SYMBOL, days=DAYS, timeframe=TIMEFRAME)

    print(f"\n=== Raw DataFrame ===")
    print(f"  Shape       : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Index type  : {type(df.index).__name__}")
    print(f"  Timezone    : {df.index.tz}")
    print(f"  Date range  : {df.index[0]}  →  {df.index[-1]}")
    print(f"\n  Last 5 bars:")
    print(df.tail(5).to_string())

    print(f"\n=== Format Checks ===")
    failures = check_format(df)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print("\nFormat does NOT match strategy expectations.")
        return

    print("  PASS  DatetimeIndex")
    print("  PASS  Timezone-aware")
    print("  PASS  Columns: open, high, low, close, volume")
    print("  PASS  No NaN values")
    print("\nFormat matches strategy expectations.")

    # Smoke-test: actually feed it into a strategy
    print(f"\n=== Strategy Smoke Test ===")
    df_y = load_bars(SYMBOL_Y, days=DAYS, timeframe=TIMEFRAME)
    shared_index = df.index.intersection(df_y.index)
    strategy = CrossSectionalStrategy(SYMBOL, SYMBOL_Y, window=30)
    strategy.set_pair_data(df.loc[shared_index], df_y.loc[shared_index])
    signals = strategy.generate_signals()
    buy_signals  = (signals["signal"] == 1).sum()
    sell_signals = (signals["signal"] == -1).sum()
    print(f"  Bars after dropna : {len(signals)}")
    print(f"  BUY  signals      : {buy_signals}")
    print(f"  SELL signals      : {sell_signals}")
    print("\nStrategy accepted the Alpaca data without modification.")


if __name__ == "__main__":
    main()
