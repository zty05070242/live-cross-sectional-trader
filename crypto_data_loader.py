"""
crypto_data_loader.py
---------------------
Fetches historical OHLCV bars for crypto pairs from Alpaca and returns
DataFrames in the same shape as alpaca_data_loader.py:

    - DatetimeIndex (timezone-aware, UTC — crypto has no Eastern session)
    - Lowercase columns: open, high, low, close, volume
    - No extra columns

Supported symbols: "BTC/USD", "ETH/USD", etc. (Alpaca crypto format)

Usage:
    from crypto_data_loader import load_crypto_bars, load_crypto_pair
    df_btc, df_eth = load_crypto_pair("BTC/USD", "ETH/USD", days=90)
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca_client import client as trading_client

# Crypto data client — no API key required for crypto, but we pass them
# for consistency and in case of rate-limit benefits.
_crypto_client = CryptoHistoricalDataClient(
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


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Drop symbol level, keep OHLCV columns, ensure UTC index."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def load_crypto_bars(symbol: str, days: int = 90, timeframe: str = "1H") -> pd.DataFrame:
    """
    Fetch historical bars for a single crypto symbol.

    Parameters
    ----------
    symbol    : Alpaca crypto pair, e.g. "BTC/USD"
    days      : number of calendar days to look back from now
    timeframe : one of "1Min", "5Min", "15Min", "30Min", "1H", "1D"

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns [open, high, low, close, volume]
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"timeframe must be one of {list(_TIMEFRAME_MAP)}, got '{timeframe}'")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_TIMEFRAME_MAP[timeframe],
        start=start,
        end=end,
    )

    bars = _crypto_client.get_crypto_bars(request)
    return _normalize(bars.df, symbol)


def load_crypto_pair(
    symbol_x: str,
    symbol_y: str,
    days: int = 90,
    timeframe: str = "1H",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch bars for two crypto symbols in a single API call and return two
    aligned DataFrames (df_x, df_y) with a shared DatetimeIndex.

    Uses an inner join so only timestamps present in both are returned —
    no forward-filling of gaps.

    Parameters
    ----------
    symbol_x  : driver symbol, e.g. "BTC/USD"
    symbol_y  : target symbol, e.g. "ETH/USD"
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

    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol_x, symbol_y],
        timeframe=_TIMEFRAME_MAP[timeframe],
        start=start,
        end=end,
    )

    bars = _crypto_client.get_crypto_bars(request)
    df_all = bars.df  # MultiIndex: (symbol, timestamp)

    df_x = _normalize(df_all.loc[symbol_x], symbol_x)
    df_y = _normalize(df_all.loc[symbol_y], symbol_y)

    # Align on shared timestamps only
    shared_index = df_x.index.intersection(df_y.index)
    return df_x.loc[shared_index], df_y.loc[shared_index]
