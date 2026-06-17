"""Property + regression tests for the sentiment-signal backtest engine.

Covers the four leakage/correctness invariants the brief requires of
:func:`wsb_sentiment.backtest.engine.run_signal_backtest`:

- **PIT restriction** — a ticker absent from the as-of universe is never traded.
- **Identical OOS index** — the signal, buy-and-hold, and attention-only baselines
  are reported on the exact same post-purge/embargo out-of-sample index.
- **Purge holds** — the OOS index starts strictly after the train/test boundary
  plus the purge+embargo gap.
- **No-lookahead** — perturbing returns at or before the OOS start cannot change
  any OOS portfolio return (positions are already lagged; nothing is refit OOS).

These tests build positions with a tiny self-contained train-only standardize /
threshold / ``shift(lag)`` helper, so they do not depend on the separately-owned
``signal.build`` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._exceptions import InsufficientDataError, ValidationError
from wsb_sentiment.backtest.engine import (
    SignalBacktestResult,
    attention_only_positions,
    run_signal_backtest,
)
from wsb_sentiment.signal.build import SignalSpec

# --------------------------------------------------------------------------- #
# Local fixtures / helpers (self-contained; no signal.build dependency)
# --------------------------------------------------------------------------- #


def _panel(seed: int, n_obs: int = 120, n_assets: int = 4) -> dict[str, pd.DataFrame]:
    """A small seeded sentiment/return/mention panel with a train/test boundary."""
    gen = np.random.default_rng(seed)
    index = pd.date_range("2021-01-04", periods=n_obs, freq="B")
    tickers = [f"TKR{i:02d}" for i in range(n_assets)]
    sentiment = pd.DataFrame(gen.standard_normal((n_obs, n_assets)), index=index, columns=tickers)
    returns = pd.DataFrame(
        gen.standard_normal((n_obs, n_assets)) * 0.01, index=index, columns=tickers
    )
    mentions = pd.DataFrame(
        gen.poisson(20.0, size=(n_obs, n_assets)).astype("float64"),
        index=index,
        columns=tickers,
    )
    return {
        "sentiment": sentiment,
        "returns": returns,
        "mentions": mentions,
        "index": index,
        "tickers": tickers,
        "train_end": index[n_obs // 2],
    }


def _build_positions(
    sentiment: pd.DataFrame, spec: SignalSpec, *, train_end: pd.Timestamp
) -> pd.DataFrame:
    """Train-only standardize -> threshold -> ``shift(lag)`` (signal.build replica)."""
    if spec.window > 1:
        smoothed = sentiment.rolling(window=spec.window, min_periods=1).mean()
    else:
        smoothed = sentiment
    train = smoothed.loc[smoothed.index <= train_end]
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0).clip(lower=1e-12)
    z = (smoothed - mean) / std
    raw = np.sign(z).where(z.abs() > spec.threshold, other=0.0)
    if spec.long_only:
        raw = raw.clip(lower=0.0)
    return raw.shift(spec.lag)


# --------------------------------------------------------------------------- #
# PIT restriction
# --------------------------------------------------------------------------- #


def test_pit_restriction_never_trades_excluded_ticker() -> None:
    """A ticker masked out of the as-of universe contributes zero P&L all OOS."""
    p = _panel(seed=1)

    # TKR03 is NEVER in the universe; give it a huge return so any leak is obvious.
    mask = pd.DataFrame(True, index=p["index"], columns=p["tickers"])
    mask["TKR03"] = False
    poisoned = p["returns"].copy()
    poisoned["TKR03"] = 5.0  # would dominate if ever traded

    spec_active = SignalSpec(threshold=-1.0)  # force every name active
    positions = _build_positions(p["sentiment"], spec_active, train_end=p["train_end"])

    result = run_signal_backtest(
        positions,
        poisoned,
        mention_count=p["mentions"],
        spec=spec_active,
        train_end=p["train_end"],
        cost_bps=0.0,
        universe_mask=mask,
    )
    # With TKR03 excluded, the enormous return must not appear: net stays bounded.
    assert result.net_returns.abs().max() < 1.0
    assert result.buyhold_returns.abs().max() < 1.0
    assert result.attention_returns.abs().max() < 1.0


def test_pit_restriction_full_universe_includes_ticker() -> None:
    """Control: WITHOUT the mask, the poisoned ticker leaks into the book."""
    p = _panel(seed=1)
    spec = SignalSpec(threshold=-1.0)
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    poisoned = p["returns"].copy()
    poisoned["TKR03"] = 5.0

    result = run_signal_backtest(
        positions,
        poisoned,
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        cost_bps=0.0,
        universe_mask=None,
    )
    # No mask => the dominating ticker is traded => large returns appear.
    assert result.net_returns.abs().max() > 0.5


def test_pit_mask_aligns_to_common_columns() -> None:
    """A mask with extra/missing tickers is reindexed (missing => not tradable)."""
    p = _panel(seed=2)
    spec = SignalSpec(threshold=-1.0)
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    # Mask only covers two tickers; the rest reindex to False (never traded).
    mask = pd.DataFrame(True, index=p["index"], columns=["TKR00", "TKR01"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        universe_mask=mask,
    )
    assert result.meta["pit_masked"] is True
    # Buy-hold holds only the two tradable names => weights sum to 1 over them.
    assert np.isfinite(result.buyhold_returns.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Identical OOS index across strategies
# --------------------------------------------------------------------------- #


def test_identical_oos_index_across_strategies() -> None:
    """Signal, buy-hold, attention, turnover all share ONE OOS index."""
    p = _panel(seed=3)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
    )
    idx = result.net_returns.index
    assert result.gross_returns.index.equals(idx)
    assert result.buyhold_returns.index.equals(idx)
    assert result.attention_returns.index.equals(idx)
    assert result.turnover.index.equals(idx)
    assert result.n_oos == len(idx)
    assert len(idx) == idx.nunique()


@pytest.mark.parametrize(("purge", "embargo"), [(0, 0), (1, 1), (2, 3), (5, 2)])
def test_identical_oos_index_under_varying_gaps(purge: int, embargo: int) -> None:
    """The shared-index invariant survives any purge/embargo configuration."""
    p = _panel(seed=4, n_obs=160)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        purge=purge,
        embargo=embargo,
    )
    idx = result.net_returns.index
    for series in (
        result.gross_returns,
        result.buyhold_returns,
        result.attention_returns,
        result.turnover,
    ):
        assert series.index.equals(idx)


# --------------------------------------------------------------------------- #
# Purge / embargo holds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("purge", "embargo"), [(0, 0), (1, 1), (3, 2), (4, 4)])
def test_purge_embargo_drops_boundary_observations(purge: int, embargo: int) -> None:
    """OOS starts strictly after train_end + (purge+embargo) dropped rows."""
    p = _panel(seed=5, n_obs=160)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        purge=purge,
        embargo=embargo,
    )
    full = p["index"].sort_values()
    post = full[full > p["train_end"]]
    expected_first = post[purge + embargo]
    assert result.net_returns.index[0] == expected_first
    # Every OOS date is strictly after the boundary.
    assert (result.net_returns.index > p["train_end"]).all()


def test_larger_gap_yields_fewer_or_equal_oos_obs() -> None:
    """Monotonicity: a wider purge+embargo gap never grows the OOS index."""
    p = _panel(seed=6, n_obs=160)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])

    def n_oos(gap: int) -> int:
        return run_signal_backtest(
            positions,
            p["returns"],
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
            purge=gap,
            embargo=0,
        ).n_oos

    assert n_oos(0) >= n_oos(2) >= n_oos(5)


def test_insufficient_oos_raises() -> None:
    """A gap that consumes the whole OOS slice raises InsufficientDataError."""
    p = _panel(seed=7, n_obs=60)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    with pytest.raises(InsufficientDataError):
        run_signal_backtest(
            positions,
            p["returns"],
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
            purge=100,
            embargo=100,
        )


# --------------------------------------------------------------------------- #
# No-lookahead
# --------------------------------------------------------------------------- #


def test_no_lookahead_perturbing_past_returns_leaves_oos_unchanged() -> None:
    """Mutating returns at/before the OOS boundary cannot change any OOS P&L.

    Positions are already ``shift(lag)``-ed and nothing is refit on the OOS slice,
    so a perturbation strictly inside the in-sample + purged region must leave
    every out-of-sample return byte-for-byte identical.
    """
    p = _panel(seed=8, n_obs=160)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])

    base = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        purge=1,
        embargo=1,
    )
    oos_start = base.net_returns.index[0]

    perturbed = p["returns"].copy()
    pre = perturbed.index < oos_start
    perturbed.loc[pre] = perturbed.loc[pre] + 99.0  # huge shock, all in the past

    after = run_signal_backtest(
        positions,
        perturbed,
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        purge=1,
        embargo=1,
    )
    pd.testing.assert_series_equal(base.net_returns, after.net_returns)
    pd.testing.assert_series_equal(base.buyhold_returns, after.buyhold_returns)
    pd.testing.assert_series_equal(base.attention_returns, after.attention_returns)


def test_no_same_bar_return_with_constant_positions() -> None:
    """A lag-1 position earns the NEXT day's return, never the same bar.

    Hand-built positions that are flat until a single day, then +1 in one name,
    must earn that name's return exactly one row later (the shift was applied
    upstream; the engine must not re-advance or look back).
    """
    index = pd.date_range("2021-01-04", periods=40, freq="B")
    tickers = ["A", "B"]
    returns = pd.DataFrame(0.0, index=index, columns=tickers)
    returns.loc[index[25], "A"] = 0.1  # the return to capture
    # Position decided for day 25 must already sit on row 25 (caller pre-shifted).
    positions = pd.DataFrame(0.0, index=index, columns=tickers)
    positions.loc[index[25], "A"] = 1.0
    mentions = pd.DataFrame(1.0, index=index, columns=tickers)
    train_end = index[20]

    result = run_signal_backtest(
        positions,
        returns,
        mention_count=mentions,
        spec=SignalSpec(),
        train_end=train_end,
        cost_bps=0.0,
        purge=0,
        embargo=0,
    )
    # The position on row 25 earns row 25's return (the upstream shift already
    # offset it from sentiment day 24); no other OOS day has P&L.
    assert result.gross_returns.loc[index[25]] == pytest.approx(0.1)
    nonzero = result.gross_returns[result.gross_returns != 0.0]
    assert list(nonzero.index) == [index[25]]


# --------------------------------------------------------------------------- #
# Costs / turnover / baselines
# --------------------------------------------------------------------------- #


def test_net_sharpe_non_increasing_in_cost() -> None:
    """Higher per-side cost never improves the net mean return (cost grid)."""
    p = _panel(seed=9, n_obs=160)
    spec = SignalSpec(threshold=-1.0)  # keep the book active => real turnover
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])

    means = []
    for bps in (0.0, 5.0, 20.0, 100.0):
        result = run_signal_backtest(
            positions,
            p["returns"],
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
            cost_bps=bps,
        )
        means.append(float(result.net_returns.mean()))
    assert means == sorted(means, reverse=True)


def test_gross_equals_net_at_zero_cost() -> None:
    """At zero cost the net and gross OOS series coincide exactly."""
    p = _panel(seed=10)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
        cost_bps=0.0,
    )
    pd.testing.assert_series_equal(result.net_returns, result.gross_returns, check_names=False)


def test_turnover_non_negative_and_bounded() -> None:
    """One-way turnover lies in [0, 1] for an L1-normalized long/short book."""
    p = _panel(seed=11)
    spec = SignalSpec(threshold=-1.0)
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
    )
    tau = result.turnover.to_numpy()
    assert (tau >= -1e-12).all()
    assert (tau <= 1.0 + 1e-9).all()


def test_attention_only_positions_are_lagged_and_bounded() -> None:
    """The attention baseline is shift(lag)-ed and lives in {-1,0,+1}."""
    p = _panel(seed=12)
    spec = SignalSpec(lag=1, threshold=0.0)
    pos = attention_only_positions(p["mentions"], spec, train_end=p["train_end"])
    # Leading ``lag`` rows are NaN from the shift.
    assert pos.iloc[: spec.lag].isna().to_numpy().all()
    valid = pos.dropna().to_numpy()
    assert set(np.unique(valid)).issubset({-1.0, 0.0, 1.0})


def test_attention_only_long_only_has_no_shorts() -> None:
    """A long-only spec yields attention positions in {0, +1}."""
    p = _panel(seed=13)
    spec = SignalSpec(long_only=True)
    pos = attention_only_positions(p["mentions"], spec, train_end=p["train_end"])
    valid = pos.dropna().to_numpy()
    assert (valid >= 0.0).all()


# --------------------------------------------------------------------------- #
# Result plumbing / validation
# --------------------------------------------------------------------------- #


def test_result_to_dict_is_json_safe() -> None:
    """``to_dict`` returns ISO-string keys and finite-or-None scalars."""
    p = _panel(seed=14)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
    )
    assert isinstance(result, SignalBacktestResult)
    d = result.to_dict()
    assert set(d) >= {
        "net_returns",
        "gross_returns",
        "buyhold_returns",
        "attention_returns",
        "turnover",
        "cost_bps",
        "n_oos",
        "meta",
    }
    for key, val in d["net_returns"].items():
        assert isinstance(key, str)
        assert val is None or np.isfinite(val)
    assert d["meta"]["oos_start"] <= d["meta"]["oos_end"]


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_negative_or_nonfinite_cost_raises(bad: float) -> None:
    """A negative or non-finite cost is rejected."""
    p = _panel(seed=15)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    with pytest.raises(ValidationError):
        run_signal_backtest(
            positions,
            p["returns"],
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
            cost_bps=bad,
        )


def test_negative_purge_or_embargo_raises() -> None:
    """Negative boundary guards are rejected."""
    p = _panel(seed=16)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    with pytest.raises(ValidationError):
        run_signal_backtest(
            positions,
            p["returns"],
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
            purge=-1,
        )


def test_no_common_dates_raises() -> None:
    """Disjoint date indexes across panels raise a ValidationError."""
    p = _panel(seed=17)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    shifted_returns = p["returns"].copy()
    shifted_returns.index = shifted_returns.index + pd.Timedelta(days=10_000)
    with pytest.raises(ValidationError):
        run_signal_backtest(
            positions,
            shifted_returns,
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
        )


def test_no_common_tickers_raises() -> None:
    """Disjoint ticker columns across panels raise a ValidationError."""
    p = _panel(seed=18)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    renamed_returns = p["returns"].copy()
    renamed_returns.columns = [f"ZZZ{i}" for i in range(renamed_returns.shape[1])]
    with pytest.raises(ValidationError):
        run_signal_backtest(
            positions,
            renamed_returns,
            mention_count=p["mentions"],
            spec=spec,
            train_end=p["train_end"],
        )


def test_to_dict_maps_nan_to_none() -> None:
    """A non-finite scalar in a result series is rendered as ``None`` by to_dict."""
    p = _panel(seed=19)
    spec = SignalSpec()
    positions = _build_positions(p["sentiment"], spec, train_end=p["train_end"])
    result = run_signal_backtest(
        positions,
        p["returns"],
        mention_count=p["mentions"],
        spec=spec,
        train_end=p["train_end"],
    )
    # Inject a NaN to exercise the _safe_float NaN branch through to_dict.
    poisoned = result.net_returns.copy()
    poisoned.iloc[0] = float("nan")
    object.__setattr__(result, "net_returns", poisoned)
    rendered = result.to_dict()["net_returns"]
    first_key = next(iter(rendered))
    assert rendered[first_key] is None
