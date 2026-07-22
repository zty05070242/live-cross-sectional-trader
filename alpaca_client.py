"""
alpaca_client.py
----------------
Single point of entry for all Alpaca API access.

Reads ALPACA_API_KEY and ALPACA_SECRET_KEY from the environment (via .env),
then builds and exposes a module-level TradingClient configured for paper trading.

Usage in other modules:
    from alpaca_client import client
    account = client.get_account()
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()  # reads .env into os.environ if present

_api_key = os.environ.get("ALPACA_API_KEY")
_secret_key = os.environ.get("ALPACA_SECRET_KEY")

if not _api_key or not _secret_key:
    raise EnvironmentError(
        "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. "
        "Copy .env.example to .env and fill in your paper trading credentials."
    )

# paper=True points all requests at Alpaca's paper trading environment.
# Flip to False only when switching to live trading.
client = TradingClient(api_key=_api_key, secret_key=_secret_key, paper=True)
