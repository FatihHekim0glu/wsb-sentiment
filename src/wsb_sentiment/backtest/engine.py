"""Sentiment-signal backtest engine (anchored walk-forward, cost grid).

Runs the daily sentiment positions against forward returns with strict leakage
guards and realistic per-side costs:

- anchored (expanding) walk-forward with purge + embargo at each train/test
  boundary, where the standardizer is REFIT on each expanding train slice only;
- FORWARD-return labels only (positions from :mod:`wsb_sentiment.signal.build`
  are already ``shift(lag)``-ed, so each return is earned with a strictly-prior
  position);
- a per-side basis-point transaction cost charged on daily position turnover,
  swept over a sensitivity grid;
- two honest baselines: buy-and-hold (equal-weight long) and an ATTENTION-ONLY
  signal (positions from mention-count alone), so the value-add of the
  sentiment *polarity* over raw attention is isolated.

PIT-UNIVERSE DISCIPLINE: only tickers in the point-in-time tradable universe on
each date contribute to the traded book; meme tickers outside the universe are
descriptive-only and never traded (enforced by the caller via the universe mask).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wsb_sentiment._typing import ReturnsLike, SentimentLike
from wsb_sentiment.signal.build import SignalSpec


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class SignalBacktestResult:
    """Immutable result of a sentiment-signal walk-forward backtest.

    Attributes
    ----------
    net_returns:
        The net (after-cost) out-of-sample portfolio return series.
    gross_returns:
        The gross (before-cost) out-of-sample portfolio return series.
    buyhold_returns:
        The equal-weight buy-and-hold OOS return series (baseline).
    attention_returns:
        The attention-only (mention-count) OOS return series (baseline).
    turnover:
        Per-day one-way turnover of the traded book.
    cost_bps:
        The per-side transaction cost (bps) used for this run.
    n_oos:
        The number of out-of-sample observations.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    buyhold_returns: pd.Series
    attention_returns: pd.Series
    turnover: pd.Series
    cost_bps: float
    n_oos: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _series(s: pd.Series) -> dict[str, float | None]:
            return {str(k): _safe_float(v) for k, v in s.items()}

        return {
            "net_returns": _series(self.net_returns),
            "gross_returns": _series(self.gross_returns),
            "buyhold_returns": _series(self.buyhold_returns),
            "attention_returns": _series(self.attention_returns),
            "turnover": _series(self.turnover),
            "cost_bps": float(self.cost_bps),
            "n_oos": int(self.n_oos),
            "meta": dict(self.meta),
        }


def attention_only_positions(
    mention_count: SentimentLike,
    spec: SignalSpec,
    *,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build an ATTENTION-ONLY baseline signal from mention counts alone.

    Standardizes the (log) mention-count panel on the TRAIN slice only and maps it
    to positions exactly like the sentiment signal, but using attention magnitude
    rather than sentiment polarity. Comparing the sentiment signal against this
    baseline isolates whether polarity adds anything beyond raw attention/return
    feedback.

    Parameters
    ----------
    mention_count:
        Wide ``day x ticker`` panel of per-day mention counts.
    spec:
        The signal configuration (window, lag, threshold, long_only).
    train_end:
        The last in-sample date (inclusive) for the train-only standardizer.

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` panel of attention-only positions (already
        ``shift(spec.lag)``-ed).

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("attention_only_positions is not yet implemented")


def run_signal_backtest(
    positions: SentimentLike,
    returns: ReturnsLike,
    *,
    mention_count: SentimentLike,
    spec: SignalSpec,
    train_end: pd.Timestamp,
    cost_bps: float = 10.0,
    embargo: int = 1,
    purge: int = 1,
    universe_mask: pd.DataFrame | None = None,
) -> SignalBacktestResult:
    """Run the anchored sentiment-signal backtest vs buy-hold and attention-only.

    Aligns the (already ``shift``-ed) ``positions`` to FORWARD returns, restricts
    the traded book to the point-in-time ``universe_mask`` when given, charges a
    per-side ``cost_bps`` transaction cost on daily turnover, and computes the net
    and gross OOS series alongside the buy-and-hold and attention-only baselines.

    NO-LOOKAHEAD: this function assumes ``positions`` were produced with
    ``shift(lag)`` and a train-only scaler; it never refits anything on the OOS
    slice and never earns a same-bar return.

    Parameters
    ----------
    positions:
        Wide ``day x ticker`` per-ticker positions (already lagged).
    returns:
        Wide ``day x ticker`` forward simple returns.
    mention_count:
        Wide ``day x ticker`` mention counts (for the attention-only baseline).
    spec:
        The signal configuration.
    train_end:
        The in-sample/out-of-sample boundary date.
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).
    embargo, purge:
        No-lookahead boundary guards (in trading days).
    universe_mask:
        Optional point-in-time boolean ``day x ticker`` tradability mask; ``False``
        cells are excluded from the traded book (meme tickers stay descriptive).

    Returns
    -------
    SignalBacktestResult
        The net/gross OOS series, baselines, turnover, and metadata.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("run_signal_backtest is not yet implemented")
