"""
strategies/cross_sectional.py
-----------------------------------
Cross-sectional mean-reversion strategy.

Fits a rolling OLS regression of y_returns ~ x_returns to discover the
"universal relationship" between two assets (e.g. BTC drives ETH).

Signal logic
------------
When the latest residual's z-score exceeds a threshold, the target asset (Y)
has deviated from where the driver asset (X) says it should be:

    z_score > +threshold  →  Y is ABOVE predicted  →  SELL  (-1)
                              (Y overshot, expect it to fall back)
    z_score < -threshold  →  Y is BELOW predicted  →  BUY   (+1)
                              (Y undershot, expect it to rise)
    |z_score| <= threshold →  no signal              HOLD   (0)

Example
-------
    BTC rises 1%.  Historical beta = 0.8.
    Predicted ETH move = +0.8%.
    Actual ETH move    = +0.4%.
    Residual = 0.4% - 0.8% = -0.4%  →  negative z-score  →  BUY ETH
    (ETH hasn't moved enough yet; expect it to catch up)
"""

import numpy as np
import pandas as pd
from strategies._strategy_base_class import Strategy


class CrossSectionalStrategy(Strategy):
    """
    Parameters
    ----------
    symbol_x    : driver asset ticker (e.g. "BTC/USD")
    symbol_y    : target asset ticker to trade (e.g. "ETH/USD")
    window      : rolling OLS lookback in bars (minimum 30, default 60)
    z_threshold : |z-score| required to generate a signal (default 1.5)
    """

    def __init__(
        self,
        symbol_x: str,
        symbol_y: str,
        window: int = 60,
        z_threshold: float = 1.5,
    ) -> None:
        super().__init__(name=f"CrossSectional({symbol_x}/{symbol_y})")
        if window < 30:
            raise ValueError(f"window must be >= 30 for stable OLS, got {window}")
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be positive, got {z_threshold}")

        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.window = window
        self.z_threshold = z_threshold

        self._signals: pd.DataFrame | None = None

    def set_pair_data(self, df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:
        """
        Load aligned OHLCV DataFrames for the driver (X) and target (Y).
        Both must share the same DatetimeIndex (use load_crypto_pair).
        Resets any previously generated signals.
        """
        if not df_x.index.equals(df_y.index):
            raise ValueError("df_x and df_y must have identical indices. Use load_crypto_pair().")
        if len(df_x) < self.window + 1:
            raise ValueError(
                f"Need at least {self.window + 1} bars to generate signals, got {len(df_x)}."
            )

        self._x_returns = df_x["close"].pct_change().dropna()
        self._y_returns = df_y["close"].pct_change().dropna()
        self._signals = None
        self._signals_generated = False

    def generate_signals(self) -> pd.DataFrame:
        """
        Run rolling OLS and compute z-scored residuals.

        Returns a DataFrame indexed by timestamp with columns:
            x_return, y_return, beta, alpha,
            predicted, residual, resid_std, z_score, signal
        """
        x = self._x_returns
        y = self._y_returns

        alpha_s, beta_s = self._rolling_ols(x, y)

        predicted = alpha_s + beta_s * x
        residual = y - predicted

        resid_std = residual.rolling(window=self.window).std()

        # Avoid division by zero in flat/early periods
        z_score = residual / resid_std.replace(0, np.nan)

        signal = pd.Series(0, index=z_score.index, dtype=int)
        signal[z_score > self.z_threshold] = -1   # Y overshot  → SELL
        signal[z_score < -self.z_threshold] = 1   # Y undershot → BUY

        self._signals = pd.DataFrame({
            "x_return":  x,
            "y_return":  y,
            "beta":      beta_s,
            "alpha":     alpha_s,
            "predicted": predicted,
            "residual":  residual,
            "resid_std": resid_std,
            "z_score":   z_score,
            "signal":    signal,
        }).dropna(subset=["z_score"])

        self._signals_generated = True
        return self._signals

    def get_latest_signal(self) -> dict:
        """
        Return the most recent bar's signal as a dict.

        Keys: symbol_x, symbol_y, timestamp, signal,
              z_score, beta, alpha, residual, resid_std
        """
        if self._signals is None or not self._signals_generated:
            raise ValueError("Call generate_signals() before get_latest_signal().")

        row = self._signals.iloc[-1]
        return {
            "symbol_x":  self.symbol_x,
            "symbol_y":  self.symbol_y,
            "timestamp": self._signals.index[-1],
            "signal":    int(row["signal"]),
            "z_score":   float(row["z_score"]),
            "beta":      float(row["beta"]),
            "alpha":     float(row["alpha"]),
            "residual":  float(row["residual"]),
            "resid_std": float(row["resid_std"]),
        }

    def get_relationship_summary(self) -> dict:
        """
        Return a human-readable summary of the fitted relationship.
        Uses the most recent OLS window's beta and alpha.
        """
        if self._signals is None:
            raise ValueError("Call generate_signals() first.")

        latest = self._signals.iloc[-1]
        beta = latest["beta"]
        direction = "same direction" if beta > 0 else "opposite direction"
        return {
            "driver":    self.symbol_x,
            "target":    self.symbol_y,
            "beta":      round(beta, 4),
            "alpha":     round(latest["alpha"], 6),
            "interpretation": (
                f"When {self.symbol_x} moves 1%, {self.symbol_y} tends to move "
                f"{beta * 100:.2f}% ({direction})"
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_ols(
        self, x: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Pure-numpy rolling OLS without statsmodels.

        For each window ending at bar t:
            beta  = cov(x_w, y_w) / var(x_w)
            alpha = mean(y_w) - beta * mean(x_w)

        Bars before the first full window are filled with NaN.
        """
        n = len(x)
        xv = x.values
        yv = y.values
        alphas = np.full(n, np.nan)
        betas  = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            x_w = xv[i - self.window + 1: i + 1]
            y_w = yv[i - self.window + 1: i + 1]
            x_mean = x_w.mean()
            y_mean = y_w.mean()
            x_dev = x_w - x_mean
            denom = np.dot(x_dev, x_dev)
            if denom == 0:
                continue   # flat x window — skip, leave NaN
            betas[i]  = np.dot(x_dev, y_w - y_mean) / denom
            alphas[i] = y_mean - betas[i] * x_mean

        return (
            pd.Series(alphas, index=x.index),
            pd.Series(betas,  index=x.index),
        )

    # Override to prevent misuse — pair strategies use set_pair_data()
    def set_data(self, data: pd.DataFrame) -> None:
        raise NotImplementedError(
            "CrossSectionalStrategy uses set_pair_data(df_x, df_y), not set_data()."
        )
