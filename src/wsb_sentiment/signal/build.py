"""Build a per-ticker daily position signal from the sentiment aggregate.

Pipeline (leakage-guarded end to end):

1. Optionally smooth the daily sentiment over a trailing ``window`` (causal).
2. Standardize with a scaler whose mean/std are FIT ON THE TRAIN SLICE ONLY
   (:func:`fit_standardizer`) and applied to the full panel — never refit on the
   OOS slice, so the standardization carries no out-of-sample information.
3. Threshold the standardized score into a position: long/short
   (``sign(z - threshold)``) or long/flat, by ticker and day.
4. Apply ``signal.shift(lag)`` (``lag >= 1``) so a position earned on day ``t``
   was decided strictly before ``t`` — the no-same-bar-lookahead guarantee.

Standardization is scale-invariant (property-tested) and the shift is
shift-equivariant; both are no-lookahead invariants enforced by the suite.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from wsb_sentiment._typing import SentimentLike


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The configuration of a single daily-sentiment signal.

    Attributes
    ----------
    window:
        Trailing (causal) smoothing window in trading days; ``1`` = no smoothing.
    lag:
        Position application lag in trading days (``>= 1``); the ``shift`` applied
        so the signal cannot see the same-bar return.
    threshold:
        Standardized-score threshold; positions activate only beyond ``|z| >
        threshold``.
    long_only:
        If ``True``, emit long/flat positions; else long/short.
    """

    window: int = 1
    lag: int = 1
    threshold: float = 0.0
    long_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StandardizerState:
    """The train-fit standardization parameters (mean/std per ticker).

    Attributes
    ----------
    mean:
        Per-ticker mean of the (smoothed) sentiment over the TRAIN slice.
    std:
        Per-ticker standard deviation over the TRAIN slice (floored at ``EPS``).
    n_train:
        The number of train observations the parameters were fit on.
    """

    mean: pd.Series
    std: pd.Series
    n_train: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this state."""
        return {
            "mean": {str(k): float(v) for k, v in self.mean.items()},
            "std": {str(k): float(v) for k, v in self.std.items()},
            "n_train": int(self.n_train),
        }


def fit_standardizer(
    sentiment: SentimentLike,
    *,
    train_end: pd.Timestamp,
    window: int = 1,
) -> StandardizerState:
    """Fit per-ticker standardization parameters on the TRAIN slice only.

    Computes the per-ticker mean and std of the (optionally ``window``-smoothed)
    sentiment over rows with index ``<= train_end`` ONLY. These parameters are
    later applied to the full panel by :func:`build_positions`, so no
    out-of-sample statistic ever enters the standardization (the train-only-scaler
    leakage guard).

    Parameters
    ----------
    sentiment:
        The wide daily per-ticker sentiment panel (rows = day, columns = ticker).
    train_end:
        The last in-sample date (inclusive); rows after it are excluded from the
        fit.
    window:
        Trailing causal smoothing window applied before fitting.

    Returns
    -------
    StandardizerState
        The per-ticker mean/std fit on the train slice.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("fit_standardizer is not yet implemented")


def build_positions(
    sentiment: SentimentLike,
    state: StandardizerState,
    spec: SignalSpec,
) -> pd.DataFrame:
    """Standardize, threshold, and shift the sentiment into per-ticker positions.

    Applies ``state`` (fit on TRAIN ONLY) to standardize the full panel, thresholds
    into long/short or long/flat positions by ``spec``, and applies
    ``shift(spec.lag)`` so a position earned on day ``t`` was decided strictly
    before ``t``. The returned frame is index-aligned to ``sentiment`` with the
    leading ``lag`` rows held flat (no position).

    Parameters
    ----------
    sentiment:
        The wide daily per-ticker sentiment panel.
    state:
        The train-fit standardization parameters from :func:`fit_standardizer`.
    spec:
        The signal configuration (window, lag, threshold, long_only).

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` panel of positions in ``{-1, 0, +1}`` (long/short)
        or ``{0, +1}`` (long/flat), already shifted by ``spec.lag``.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("build_positions is not yet implemented")
