"""
alpaca_data_loader.py
---------------------
Fetches historical OHLCV bars from Alpaca and returns a DataFrame
in the same shape as the backtester's data_loader.py (yfinance):

    - DatetimeIndex (timezone-aware, US/Eastern)
    - Lowercase columns: open, high, low, close, volume
    - No extra columns (vwap, trade_count are dropped)

Usage:
    from alpaca_data_loader import load_bars
    df = load_bars("SPY", days=30, timeframe="15Min")
    strategy.set_data(df)
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca_client import client as trading_client   # reuse auth credentials

# Historical data client — uses the same keys as the trading client.
# We pull them from the already-validated trading client rather than re-reading .env.
_hd_client = StockHistoricalDataClient(
    api_key=trading_client._api_key,
    secret_key=trading_client._secret_key,
)

_TIMEFRAME_MAP = {
    "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
    "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1H":    TimeFrame(1,  TimeFrameUnit.Hour),
    "1D":    TimeFrame(1,  TimeFrameUnit.Day),
}


def load_bars(symbol: str, days: int = 30, timeframe: str = "15Min") -> pd.DataFrame:
    """
    Fetch historical bars for a single symbol.

    Parameters
    ----------
    symbol    : ticker string, e.g. "SPY"
    days      : number of calendar days to look back from now
    timeframe : one of "1Min", "5Min", "15Min", "30Min", "1H", "1D"

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"timeframe must be one of {list(_TIMEFRAME_MAP)}, got '{timeframe}'")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_TIMEFRAME_MAP[timeframe],
        start=start,
        end=end,
        feed=DataFeed.IEX,   # free-tier feed; upgrade to SIP with a paid subscription
    )

    bars = _hd_client.get_stock_bars(request)
    df = bars.df  # MultiIndex: (symbol, timestamp)

    # Drop the symbol level — strategies expect a plain DatetimeIndex
    df = df.droplevel("symbol")

    # Keep only the columns strategies need; rename to lowercase to match yfinance output
    df = df[["open", "high", "low", "close", "volume"]]

    # Alpaca returns UTC; convert to US/Eastern to match market convention
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "timestamp"

    return df


def load_stock_pair(
    symbol_x: str,
    symbol_y: str,
    days: int = 30,
    timeframe: str = "15Min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch bars for two equities in a single API call and return two
    aligned DataFrames (df_x, df_y) with a shared DatetimeIndex.

    Uses an inner join so only timestamps present in both are returned —
    no forward-filling of gaps.

    Parameters
    ----------
    symbol_x  : driver symbol, e.g. "MU"
    symbol_y  : target symbol, e.g. "WDC"
    days      : calendar days to look back
    timeframe : bar size

    Returns
    -------
    (df_x, df_y) — two DataFrames with identical DatetimeIndex
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"timeframe must be one of {list(_TIMEFRAME_MAP)}, got '{timeframe}'")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol_x, symbol_y],
        timeframe=_TIMEFRAME_MAP[timeframe],
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = _hd_client.get_stock_bars(request)
    df_all = bars.df  # MultiIndex: (symbol, timestamp)

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "timestamp"
        return df

    df_x = _normalize(df_all.loc[symbol_x])
    df_y = _normalize(df_all.loc[symbol_y])

    # Align on shared timestamps only
    shared_index = df_x.index.intersection(df_y.index)
    return df_x.loc[shared_index], df_y.loc[shared_index]
