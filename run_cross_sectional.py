"""
run_cross_sectional.py
----------------------
Live trading loop for the BTC/ETH cross-sectional mean-reversion strategy.

How it works
------------
1. Load the last LOOKBACK_DAYS of hourly BTC/USD and ETH/USD bars.
2. Fit a rolling OLS to discover the current BTC→ETH relationship.
3. Check the latest z-score of the residual:
     z > +Z_THRESHOLD  → ETH overshot BTC  → SELL ETH
     z < -Z_THRESHOLD  → ETH undershot BTC → BUY  ETH
4. Size the position using risk % of current account equity.
5. Sleep LOOP_INTERVAL seconds and repeat.

Position management
-------------------
- Only one position in ETH at a time.
- If the signal reverses while a position is open, close first then re-enter.
- If the signal returns to 0, close the position (mean reversion complete).

Run
---
    python run_cross_sectional.py
"""

import logging
import time

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from alpaca_client import client
from crypto_data_loader import load_crypto_pair
from strategies.cross_sectional import CrossSectionalStrategy
from position_sizer import calculate_position_size

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOL_X      = "BTC/USD"   # driver  — NOT traded
SYMBOL_Y      = "ETH/USD"   # target  — this is what we buy / sell
LOOKBACK_DAYS = 120          # calendar days of history to load
TIMEFRAME     = "1H"         # bar size for OLS fitting
OLS_WINDOW    = 60           # bars in rolling OLS window (~2.5 days of hourly)
Z_THRESHOLD   = 1.5          # |z-score| to trigger entry
RISK_PCT      = 0.01         # fraction of equity to risk per trade (1%)
STOP_Z_MULT   = 2.0          # stop loss at this many residual-stds away
LOOP_INTERVAL = 60 * 60      # seconds between checks (1 hour matches bar size)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_equity() -> float:
    return float(client.get_account().equity)


def get_position_qty(symbol: str) -> float:
    """Return signed quantity held in symbol, or 0.0 if no position."""
    try:
        pos = client.get_open_position(symbol)
        return float(pos.qty_available)
    except APIError:
        return 0.0


def place_crypto_order(symbol: str, qty: float, side: str):
    """Place a GTC market order for a crypto symbol."""
    request = MarketOrderRequest(
        symbol=symbol,
        qty=round(qty, 6),
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.GTC,   # crypto requires GTC, not DAY
    )
    try:
        order = client.submit_order(request)
        log.info("Order submitted: %s %s %.6f  id=%s", side.upper(), symbol, qty, order.id)
        return order
    except APIError as e:
        raise RuntimeError(f"Failed to place {side} order for {qty} {symbol}: {e}") from e


def close_crypto_position(symbol: str):
    """Close the entire open position for a crypto symbol."""
    try:
        order = client.close_position(symbol)
        log.info("Closed position: %s", symbol)
        return order
    except APIError as e:
        raise RuntimeError(f"Failed to close {symbol}: {e}") from e


def compute_stop_price(entry_price: float, resid_std: float, side: str) -> float:
    """
    Convert residual std (in return units) to a price-level stop.

    Stop is placed at STOP_Z_MULT standard deviations against the trade.
    """
    stop_distance = entry_price * resid_std * STOP_Z_MULT
    if side == "buy":
        return entry_price - stop_distance
    return entry_price + stop_distance


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_once() -> None:
    log.info("── Checking signal ──────────────────────────────────────────────")

    # 1. Load data and fit strategy
    df_btc, df_eth = load_crypto_pair(SYMBOL_X, SYMBOL_Y, days=LOOKBACK_DAYS, timeframe=TIMEFRAME)
    strategy = CrossSectionalStrategy(SYMBOL_X, SYMBOL_Y, window=OLS_WINDOW, z_threshold=Z_THRESHOLD)
    strategy.set_pair_data(df_btc, df_eth)
    strategy.generate_signals()

    sig = strategy.get_latest_signal()
    rel = strategy.get_relationship_summary()

    log.info("Relationship: %s", rel["interpretation"])
    log.info(
        "Signal=%+d  z=%.3f  beta=%.4f  residual=%.6f  resid_std=%.6f",
        sig["signal"], sig["z_score"], sig["beta"], sig["residual"], sig["resid_std"],
    )

    # 2. Current position
    current_qty = get_position_qty(SYMBOL_Y)
    has_long  = current_qty > 0
    has_short = current_qty < 0

    new_signal = sig["signal"]   # +1, -1, or 0

    # 3. Exit logic
    if has_long and new_signal <= 0:
        log.info("Closing LONG (signal changed to %+d)", new_signal)
        close_crypto_position(SYMBOL_Y)
        has_long = False

    elif has_short and new_signal >= 0:
        log.info("Closing SHORT (signal changed to %+d)", new_signal)
        close_crypto_position(SYMBOL_Y)
        has_short = False

    # 4. Entry logic
    if new_signal == 0 or (new_signal == 1 and has_long) or (new_signal == -1 and has_short):
        log.info("No new entry needed.")
        return

    equity       = get_equity()
    entry_price  = float(df_eth["close"].iloc[-1])
    stop_price   = compute_stop_price(entry_price, sig["resid_std"], "buy" if new_signal == 1 else "sell")

    try:
        sizing = calculate_position_size(
            account_balance=equity,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            risk_pct=RISK_PCT,
        )
    except ValueError as e:
        log.warning("Position sizing rejected: %s — skipping entry.", e)
        return

    qty  = sizing["units_to_trade"]
    side = "buy" if new_signal == 1 else "sell"

    log.info(
        "Entry: %s %.6f %s  entry=%.2f  stop=%.2f  risk=$%.2f",
        side.upper(), qty, SYMBOL_Y, entry_price, stop_price, sizing["max_risk_allowed"],
    )
    place_crypto_order(SYMBOL_Y, qty, side)


def main(loop: bool = True) -> None:
    log.info("Cross-Sectional Trader starting")
    log.info("Driver: %s | Target: %s | Timeframe: %s | Window: %d bars | Z-threshold: %.1f",
             SYMBOL_X, SYMBOL_Y, TIMEFRAME, OLS_WINDOW, Z_THRESHOLD)

    while True:
        try:
            run_once()
        except Exception as e:
            log.error("run_once() failed: %s", e, exc_info=True)

        if not loop:
            break

        log.info("Sleeping %d seconds until next check.", LOOP_INTERVAL)
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
