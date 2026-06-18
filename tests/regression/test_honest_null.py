"""Regression tests pinning the honest-null behaviour and the verdict truth table.

The headline result is a credible WEAK/NEGATIVE one: a naive WSB sentiment signal
shows mild in-sample correlation that DECAYS out-of-sample and fails the Deflated
Sharpe + cost hurdles, so the derived ``signal_has_edge`` reads ``False``.

These tests build a leakage-free swept grid (window x lag x threshold x cost) over
the seeded ``decaying_signal`` and ``pure_noise`` fixtures, feed the resulting OOS
trial returns through :func:`compute_honest_stats`, and assert the pure
:func:`derive_verdict` returns ``False``. They also pin the full verdict truth
table so a single flipped hurdle can never silently yield a spurious edge.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from tests.conftest import DecayingSignal
from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.evaluation.stats import compute_honest_stats
from wsb_sentiment.evaluation.verdict import derive_verdict

# Swept grid: aggregation window x signal lag x entry threshold x per-side cost.
_WINDOWS = (1, 3, 5)
_LAGS = (1, 2)
_THRESHOLDS = (0.0, 0.25, 0.5)
_COSTS_BPS = (5.0, 10.0, 20.0)


def _trial_returns(signal: DecayingSignal) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build a leakage-free trial-return grid plus a selected net and buy-hold series.

    For each configuration the cross-sectional position is the lagged, rolling-mean
    sentiment thresholded to long/flat, the same-period payoff uses FORWARD returns
    only (the position formed on ``t`` earns the return realised at ``t+1`` already
    baked into the ``returns`` panel), and a per-side bps cost is charged on
    turnover. No same-bar information enters any position.
    """
    sentiment = signal.sentiment
    returns = signal.returns
    columns: dict[str, pd.Series] = {}

    for window, lag, threshold, cost_bps in product(_WINDOWS, _LAGS, _THRESHOLDS, _COSTS_BPS):
        rolled = sentiment.rolling(window=window, min_periods=window).mean()
        # signal.shift(lag): the position on date t uses only information through
        # t-lag, never the contemporaneous bar.
        position = (rolled.shift(lag) > threshold).astype("float64")
        # Per-side cost on position changes (turnover), in return units.
        turnover = position.diff().abs().fillna(0.0)
        cost = turnover * (cost_bps * 1e-4)
        gross = position * returns
        net = (gross - cost).mean(axis=1)  # equal-weight portfolio across tickers
        columns[f"w{window}_l{lag}_t{threshold}_c{cost_bps}"] = net

    grid = pd.DataFrame(columns).dropna(how="any")

    # The "selected" configuration is the one with the highest IN-SAMPLE Sharpe
    # (deliberately overfit on the training half) - exactly the selection bias the
    # DSR/PBO are meant to penalise. We then evaluate it OUT-OF-SAMPLE.
    train = grid.loc[grid.index <= signal.train_end]
    test = grid.loc[grid.index > signal.train_end]
    is_sharpe = train.mean() / train.std(ddof=1).replace(0.0, np.nan)
    best = str(is_sharpe.idxmax())

    net_oos = test[best]
    # Equal-weight buy-and-hold over the same OOS window.
    buyhold_oos = returns.loc[test.index].mean(axis=1)
    return test, net_oos, buyhold_oos


@pytest.mark.regression
def test_decaying_signal_has_no_oos_edge(decaying_signal: DecayingSignal) -> None:
    """The decaying signal fails the DSR/cost hurdles -> signal_has_edge is False."""
    trials, net_oos, buyhold_oos = _trial_returns(decaying_signal)
    stats = compute_honest_stats(net_oos, buyhold_oos, trials, n_grid_trials=trials.shape[1])

    # The multiplicity-adjusted Deflated Sharpe must NOT clear the credibility bar
    # on a signal whose in-sample edge decays out of sample.
    assert stats.deflated_sharpe < 0.95

    verdict = derive_verdict(stats.net_sharpe, stats.deflated_sharpe, stats.pbo, stats.hac_pvalue)
    assert verdict.signal_has_edge is False
    # The verdict carries one human-readable reason per hurdle.
    assert len(verdict.reasons) == 4


@pytest.mark.regression
def test_pure_noise_has_no_oos_edge(pure_noise: DecayingSignal) -> None:
    """Independent sentiment/returns trivially fail every edge hurdle."""
    trials, net_oos, buyhold_oos = _trial_returns(pure_noise)
    stats = compute_honest_stats(net_oos, buyhold_oos, trials, n_grid_trials=trials.shape[1])
    verdict = derive_verdict(stats.net_sharpe, stats.deflated_sharpe, stats.pbo, stats.hac_pvalue)
    assert verdict.signal_has_edge is False


@pytest.mark.regression
def test_effective_trials_below_raw_grid_on_decaying_signal(
    decaying_signal: DecayingSignal,
) -> None:
    """The correlated sweep deflates: effective trials < raw grid size."""
    trials, net_oos, buyhold_oos = _trial_returns(decaying_signal)
    stats = compute_honest_stats(net_oos, buyhold_oos, trials, n_grid_trials=trials.shape[1])
    # Many configurations produce near-identical streams, so PCA must report
    # materially fewer independent trials than the raw grid.
    assert 1.0 <= stats.n_effective_trials < trials.shape[1]


@pytest.mark.regression
@pytest.mark.parametrize(
    ("sharpe", "dsr", "pbo", "hac_p", "expected"),
    [
        # Only the all-pass row yields an edge.
        (0.80, 0.99, 0.10, 0.01, True),
        # Each single failing hurdle flips the verdict to False.
        (0.00, 0.99, 0.10, 0.01, False),  # non-positive Sharpe
        (-0.5, 0.99, 0.10, 0.01, False),  # negative Sharpe
        (0.80, 0.95, 0.10, 0.01, False),  # DSR exactly at threshold (not >)
        (0.80, 0.50, 0.10, 0.01, False),  # DSR well below threshold
        (0.80, 0.99, 0.50, 0.01, False),  # PBO exactly at threshold (not <)
        (0.80, 0.99, 0.90, 0.01, False),  # PBO high
        (0.80, 0.99, 0.10, 0.05, False),  # HAC p exactly at alpha (not <)
        (0.80, 0.99, 0.10, 0.30, False),  # HAC insignificant
        # Multiple failures still False.
        (-0.1, 0.20, 0.80, 0.40, False),
    ],
)
def test_verdict_truth_table(
    sharpe: float, dsr: float, pbo: float, hac_p: float, expected: bool
) -> None:
    """The pure verdict is True iff ALL four hurdles pass simultaneously."""
    verdict = derive_verdict(sharpe, dsr, pbo, hac_p)
    assert verdict.signal_has_edge is expected
    # Boundary discipline: thresholds are strict, so 'exactly at' never passes.
    assert verdict.oos_net_sharpe == pytest.approx(sharpe)


@pytest.mark.regression
def test_verdict_rejects_non_finite_inputs() -> None:
    """A non-finite statistic can never be coerced into an edge claim."""
    with pytest.raises(ValidationError, match="finite"):
        derive_verdict(float("nan"), 0.99, 0.10, 0.01)
    with pytest.raises(ValidationError, match="finite"):
        derive_verdict(0.8, float("inf"), 0.10, 0.01)
