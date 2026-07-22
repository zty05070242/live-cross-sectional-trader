"""
test_order_executor.py
----------------------
End-to-end test of order execution against the Alpaca paper trading account.

Steps:
  1. Verify paper-trading guard is active
  2. Place a market buy order for 1 share of SPY
  3. Poll until the order fills (or timeout)
  4. List open positions — SPY should appear
  5. Close the SPY position
  6. Confirm the position is gone

Run with:
    python test_order_executor.py
"""

import time
from alpaca.trading.enums import OrderStatus
from order_executor import (
    place_market_order,
    close_position,
    get_open_positions,
    get_order_status,
)
from alpaca.common.enums import BaseURL
from alpaca_client import client

SYMBOL = "SPY"
QTY    = 1

POLL_INTERVAL_S = 2
POLL_TIMEOUT_S  = 30   # market orders fill near-instantly in paper trading


def print_positions(positions: list):
    if not positions:
        print("  (none)")
        return
    for p in positions:
        print(
            f"  {p.symbol:6}  qty={p.qty:>6}  side={p.side.value:5}  "
            f"market_value=${float(p.market_value):>10,.2f}  "
            f"unrealized_pl=${float(p.unrealized_pl):>+,.2f}"
        )


def wait_for_fill(order_id: str) -> OrderStatus:
    """Poll order status until filled, rejected, or timeout."""
    terminal = {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED}
    elapsed = 0
    while elapsed < POLL_TIMEOUT_S:
        status = get_order_status(order_id)
        print(f"  [{elapsed:>2}s] Order status: {status.value}")
        if status in terminal:
            return status
        time.sleep(POLL_INTERVAL_S)
        elapsed += POLL_INTERVAL_S
    return get_order_status(order_id)   # final read after timeout


def main():
    # -----------------------------------------------------------------------
    # Step 1: Confirm paper guard
    # -----------------------------------------------------------------------
    print("=== Step 1: Paper Trading Guard ===")
    base_url = client._base_url.value
    assert "paper-api" in base_url, f"Unexpected base URL: {base_url}"
    print(f"  Connected to: {base_url}  ✓")

    # -----------------------------------------------------------------------
    # Step 2: Place a market buy
    # -----------------------------------------------------------------------
    print(f"\n=== Step 2: Place Market BUY — {QTY} share(s) of {SYMBOL} ===")
    order = place_market_order(SYMBOL, QTY, "buy")
    print(f"  Order ID  : {order.id}")
    print(f"  Status    : {order.status.value}")
    print(f"  Side      : {order.side.value}")
    print(f"  Qty       : {order.qty}")

    # -----------------------------------------------------------------------
    # Step 3: Poll until filled
    # -----------------------------------------------------------------------
    print(f"\n=== Step 3: Wait for Fill (timeout={POLL_TIMEOUT_S}s) ===")
    final_status = wait_for_fill(str(order.id))
    if final_status != OrderStatus.FILLED:
        print(f"\n  Order did not fill — final status: {final_status.value}")
        print("  Aborting test to avoid leaving a dangling order.")
        return
    print(f"  Filled  ✓")

    # -----------------------------------------------------------------------
    # Step 4: List positions
    # -----------------------------------------------------------------------
    print(f"\n=== Step 4: Open Positions ===")
    positions = get_open_positions()
    print_positions(positions)
    spy_held = any(p.symbol == SYMBOL for p in positions)
    if not spy_held:
        print(f"  WARNING: {SYMBOL} not found in positions after fill.")

    # -----------------------------------------------------------------------
    # Step 5: Close the position
    # -----------------------------------------------------------------------
    print(f"\n=== Step 5: Close {SYMBOL} Position ===")
    close_order = close_position(SYMBOL)
    print(f"  Close order ID  : {close_order.id}")
    print(f"  Close order side: {close_order.side.value}")

    # -----------------------------------------------------------------------
    # Step 6: Confirm closed
    # -----------------------------------------------------------------------
    print(f"\n=== Step 6: Confirm Position Closed ===")
    # Give Alpaca a moment to process the close
    time.sleep(3)
    positions_after = get_open_positions()
    spy_still_open = any(p.symbol == SYMBOL for p in positions_after)
    if spy_still_open:
        print(f"  WARNING: {SYMBOL} still appears in positions — close may be pending.")
    else:
        print(f"  {SYMBOL} position confirmed closed  ✓")

    print(f"\n  Remaining open positions: {len(positions_after)}")
    print_positions(positions_after)

    print("\n=== All steps complete ===")


if __name__ == "__main__":
    main()
