"""
test_connection.py
------------------
Verifies the Alpaca paper trading connection is working.

Run with:
    python test_connection.py

Prints account balance, buying power, and any open positions.
"""

from alpaca_client import client
from alpaca.trading.requests import GetAssetsRequest

def main():
    account = client.get_account()

    print("=== Alpaca Paper Account ===")
    print(f"  Status       : {account.status}")
    print(f"  Equity       : ${float(account.equity):,.2f}")
    print(f"  Cash         : ${float(account.cash):,.2f}")
    print(f"  Buying power : ${float(account.buying_power):,.2f}")

    positions = client.get_all_positions()
    print(f"\n=== Open Positions ({len(positions)}) ===")
    if not positions:
        print("  No open positions.")
    else:
        for pos in positions:
            print(
                f"  {pos.symbol:6}  qty={pos.qty:>8}  "
                f"side={pos.side.value}  "
                f"market_value=${float(pos.market_value):>10,.2f}  "
                f"unrealized_pl=${float(pos.unrealized_pl):>+,.2f}"
            )

if __name__ == "__main__":
    main()
