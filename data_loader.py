"""
data_loader.py
--------------
Stub for a future backtester data source.
The live trader uses alpaca_data_loader.load_bars() instead.
This function will be wired up when the backtester data source is needed.
"""

import pandas as pd


def load_historical_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    raise NotImplementedError(
        "data_loader.load_historical_data() is not implemented in the live trader. "
        "Use alpaca_data_loader.load_bars() instead."
    )
