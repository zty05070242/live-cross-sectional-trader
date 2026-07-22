"""
order_executor.py
-----------------
Thin wrapper around Alpaca's TradingClient for placing and managing orders.

All functions use the shared client from alpaca_client.py.
A paper-trading guard runs at import time — importing this module against a
live-trading client raises RuntimeError immediately.

Public API
----------
    place_market_order(symbol, qty, side)  -> Order
    close_position(symbol)                 -> Order
    get_open_positions()                   -> list[Position]
    get_order_status(order_id)             -> OrderStatus
"""

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.common.exceptions import APIError
from alpaca.common.enums import BaseURL
from alpaca_client import client

# ---------------------------------------------------------------------------
# Paper-trading guard
# ---------------------------------------------------------------------------
if client._base_url != BaseURL.TRADING_PAPER:
    raise RuntimeError(
        f"order_executor.py must only be used with the paper trading environment.\n"
        f"Client is pointed at: {client._base_url.value}\n"
        f"Expected:             {BaseURL.TRADING_PAPER.value}\n"
        "Set paper=True in alpaca_client.py."
    )

_SIDE_MAP = {
    "buy":  OrderSide.BUY,
    "sell": OrderSide.SELL,
}


def place_market_order(symbol: str, qty: float, side: str):
    """
    Place a market order.

    Parameters
    ----------
    symbol : ticker, e.g. "SPY"
    qty    : number of shares (fractional allowed on Alpaca paper)
    side   : "buy" or "sell"

    Returns
    -------
    alpaca.trading.models.Order
    """
    if side not in _SIDE_MAP:
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")

    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=_SIDE_MAP[side],
        time_in_force=TimeInForce.DAY,
    )

    try:
        order = client.submit_order(request)
    except APIError as e:
        raise RuntimeError(f"Failed to place {side} order for {qty} {symbol}: {e}") from e

    return order


def close_position(symbol: str):
    """
    Close the entire open position for a symbol at market.

    Returns the closing Order, or raises RuntimeError if there is no position
    or the API call fails.
    """
    try:
        order = client.close_position(symbol)
    except APIError as e:
        raise RuntimeError(f"Failed to close position for {symbol}: {e}") from e

    return order


def get_open_positions() -> list:
    """
    Return a list of all currently open positions.
    Each element is an alpaca.trading.models.Position.
    Returns an empty list if none are held.
    """
    try:
        return client.get_all_positions()
    except APIError as e:
        raise RuntimeError(f"Failed to fetch open positions: {e}") from e


def get_order_status(order_id: str) -> OrderStatus:
    """
    Return the current status of an order by its ID string.

    Common statuses: filled, partially_filled, new, pending_new, rejected, canceled.
    """
    try:
        order = client.get_order_by_id(order_id)
    except APIError as e:
        raise RuntimeError(f"Failed to fetch order {order_id}: {e}") from e

    return order.status
