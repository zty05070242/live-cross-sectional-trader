"""
run_memory_sector.py
---------------------
Live trading loop for the memory-sector cross-sectional strategy.

Driver: MU (Micron) — largest pure-play US memory maker, used as the
leading indicator. Each target below is paired independently against MU
using the same rolling-OLS residual z-score logic as run_cross_sectional.py
(see strategies/cross_sectional.py), just run as a basket of parallel
pair-strategies sharing one driver.

Targets: WDC, SNDK, STX, SKHY.
(SMSN / Samsung Electronics has no tradable US listing on Alpaca and is
excluded.)

Position management
--------------------
- One position per target symbol at a time.
- If a target's signal reverses while a position is open, close first
  then re-enter.
- If a target's signal returns to 0, close (mean reversion complete).
- Risk per trade = RISK_PCT of equity, split evenly across the number of
  targets so total basket risk stays bounded regardless of how many
  targets fire at once.

Run
---
    python run_memory_sector.py
"""

import logging
import time

from alpaca.common.exceptions import APIError

from alpaca_client import client
from alpaca_data_loader import load_stock_pair
from strategies.cross_sectional import CrossSectionalStrategy
from position_sizer import calculate_position_size
from order_executor import place_market_order, close_position

# ── Configuration ─────────────────────────────────────────────────────────────
DRIVER        = "MU"                          # leading indicator — NOT traded
TARGETS       = ["WDC", "SNDK", "STX", "SKHY"]  # traded independently vs DRIVER
LOOKBACK_DAYS = 13           # calendar days of history to load (intraday feed depth)
TIMEFRAME     = "5Min"       # bar size for OLS fitting — day-trading resolution
OLS_WINDOW    = 30           # bars in rolling OLS window (~2.5 hours of 5Min bars)
Z_THRESHOLD   = 1.5          # |z-score| to trigger entry
RISK_PCT      = 0.01         # total basket risk budget (fraction of equity)
STOP_Z_MULT   = 2.0          # stop loss at this many residual-stds away
LOOP_INTERVAL = 5 * 60       # seconds between checks — matches bar size

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


def compute_stop_price(entry_price: float, resid_std: float, side: str) -> float:
    """Convert residual std (in return units) to a price-level stop."""
    stop_distance = entry_price * resid_std * STOP_Z_MULT
    if side == "buy":
        return entry_price - stop_distance
    return entry_price + stop_distance


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_once_for_target(symbol_y: str, equity: float, risk_pct: float) -> None:
    log.info("── %s vs %s ──────────────────────────────", symbol_y, DRIVER)

    df_driver, df_target = load_stock_pair(DRIVER, symbol_y, days=LOOKBACK_DAYS, timeframe=TIMEFRAME)
    strategy = CrossSectionalStrategy(DRIVER, symbol_y, window=OLS_WINDOW, z_threshold=Z_THRESHOLD)
    strategy.set_pair_data(df_driver, df_target)
    strategy.generate_signals()

    sig = strategy.get_latest_signal()
    log.info(
        "Signal=%+d  z=%.3f  beta=%.4f  residual=%.6f  resid_std=%.6f",
        sig["signal"], sig["z_score"], sig["beta"], sig["residual"], sig["resid_std"],
    )

    current_qty = get_position_qty(symbol_y)
    has_long  = current_qty > 0
    has_short = current_qty < 0
    new_signal = sig["signal"]   # +1, -1, or 0

    # Exit logic — close only. Re-entry (including reversal to the opposite
    # side) is deferred to a later cycle, once the close has actually filled
    # and the shares are no longer held against the pending close order.
    if has_long and new_signal <= 0:
        log.info("Closing LONG %s (signal changed to %+d)", symbol_y, new_signal)
        close_position(symbol_y)
        return
    if has_short and new_signal >= 0:
        log.info("Closing SHORT %s (signal changed to %+d)", symbol_y, new_signal)
        close_position(symbol_y)
        return

    # Entry logic
    if new_signal == 0 or (new_signal == 1 and has_long) or (new_signal == -1 and has_short):
        log.info("No new entry needed for %s.", symbol_y)
        return

    entry_price = float(df_target["close"].iloc[-1])
    side = "buy" if new_signal == 1 else "sell"
    stop_price = compute_stop_price(entry_price, sig["resid_std"], side)

    try:
        sizing = calculate_position_size(
            account_balance=equity,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            risk_pct=risk_pct,
        )
    except ValueError as e:
        log.warning("%s sizing rejected: %s — skipping entry.", symbol_y, e)
        return

    # Short sales require whole shares on Alpaca — round for both sides for consistency.
    qty = round(sizing["units_to_trade"])
    if qty < 1:
        log.warning("%s computed qty rounds to 0 shares — skipping entry.", symbol_y)
        return

    log.info(
        "Entry: %s %d %s  entry=%.2f  stop=%.2f  risk=$%.2f",
        side.upper(), qty, symbol_y, entry_price, stop_price, sizing["max_risk_allowed"],
    )
    place_market_order(symbol_y, qty, side)


def run_once() -> None:
    equity = get_equity()
    per_target_risk = RISK_PCT / len(TARGETS)

    for symbol_y in TARGETS:
        try:
            run_once_for_target(symbol_y, equity, per_target_risk)
        except Exception as e:
            log.error("%s failed: %s", symbol_y, e, exc_info=True)


def main(loop: bool = True) -> None:
    log.info("Memory-Sector Cross-Sectional Trader starting")
    log.info("Driver: %s | Targets: %s | Timeframe: %s | Window: %d bars | Z-threshold: %.1f",
             DRIVER, ", ".join(TARGETS), TIMEFRAME, OLS_WINDOW, Z_THRESHOLD)

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
