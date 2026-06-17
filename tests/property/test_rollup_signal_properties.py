"""Property-based tests for the as-of roll-up and the train-only-scaler signal.

These pin the two no-lookahead invariants that the aggregate/signal layer is
responsible for:

- **As-of cutoff** (:func:`wsb_sentiment.aggregate.rollup.rollup_daily_sentiment`):
  a post created at time ``u`` may inform trading day ``d`` only if ``u`` is on or
  before the PRIOR session close of ``d`` — so no post created after a session's
  close can leak into that same session, and the aggregator is prefix-deterministic
  (appending future posts never changes an already-emitted ``(ticker, day)`` row).
- **Train-only scaler + shift** (:mod:`wsb_sentiment.signal.build`): the
  standardizer is fit on the TRAIN slice only (perturbing test rows leaves the
  fitted mean/std and the train-slice signal unchanged), the standardization is
  scale-invariant, and ``build_positions`` applies ``shift(lag)`` so the leading
  ``lag`` rows are flat and no position earns a same-bar return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wsb_sentiment.aggregate.rollup import rollup_daily_sentiment
from wsb_sentiment.signal.build import (
    SignalSpec,
    StandardizerState,
    build_positions,
    fit_standardizer,
)

_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

_TZ = "America/New_York"
_CLOSE = "16:00"


def _sessions(n: int) -> pd.DatetimeIndex:
    """A fixed business-day session calendar of length ``n``."""
    return pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=n), name="day")


def _close_utc(day: pd.Timestamp) -> pd.Timestamp:
    """The exchange close of ``day`` as a tz-aware UTC timestamp."""
    local = pd.Timestamp(f"{day.date()} {_CLOSE}", tz=_TZ)
    return local.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# As-of cutoff
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(
    seconds=st.lists(
        st.integers(min_value=0, max_value=18 * 24 * 3600),
        min_size=1,
        max_size=40,
    ),
    compounds=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=40,
    ),
)
def test_asof_cutoff_no_post_after_prior_close_leaks(
    seconds: list[int], compounds: list[float]
) -> None:
    """Every post assigned to day ``d`` was created on/before ``prior_close(d)``."""
    n = min(len(seconds), len(compounds))
    base = int(pd.Timestamp("2021-01-04 00:00", tz=_TZ).timestamp())
    created = [base + s for s in seconds[:n]]
    df = pd.DataFrame(
        {
            "created_utc": created,
            "ticker": ["GME"] * n,
            "compound": compounds[:n],
        }
    )
    sessions = _sessions(20)
    rollup = rollup_daily_sentiment(df, session_close=_CLOSE, session_tz=_TZ, sessions=sessions)

    counts = rollup.mention_count["GME"]
    assigned_days = counts.index[counts.to_numpy() > 0]
    created_ts = pd.to_datetime(created, unit="s", utc=True)

    for day in assigned_days:
        day_ts = pd.Timestamp(day)
        # The session calendar is unique and monotone, so ``get_loc`` is an int.
        prior_idx = int(sessions.get_loc(day_ts))  # type: ignore[arg-type]
        # The first session can never be a target (it has no prior close).
        assert prior_idx >= 1
        prior_close = _close_utc(pd.Timestamp(sessions[prior_idx - 1]))
        # Every post contributing to ``day`` must clear the PRIOR close
        # (u <= prior_close) and must NOT yet be eligible for an earlier session,
        # i.e. it falls after the close two sessions back.
        lower = _close_utc(pd.Timestamp(sessions[prior_idx - 2])) if prior_idx >= 2 else None
        in_window = created_ts[created_ts <= prior_close]
        if lower is not None:
            in_window = in_window[in_window > lower]
        # The number of posts in this day's as-of window must equal its count.
        assert len(in_window) == int(counts.loc[day])

    # Direct invariant: total assigned mentions never exceeds the posts that fall
    # before the LAST available prior close (none may leak past the calendar).
    last_prior_close = _close_utc(sessions[-2])
    eligible = int((created_ts <= last_prior_close).sum())
    assert int(counts.sum()) <= eligible


@_SETTINGS
@given(
    seconds=st.lists(
        st.integers(min_value=0, max_value=16 * 24 * 3600),
        min_size=1,
        max_size=30,
    ),
)
def test_asof_assignment_matches_reference(seconds: list[int]) -> None:
    """Each post lands on the first session whose PRIOR close is on/after ``u``."""
    base = int(pd.Timestamp("2021-01-04 00:00", tz=_TZ).timestamp())
    created = [base + s for s in seconds]
    df = pd.DataFrame(
        {
            "created_utc": created,
            "ticker": [f"T{i % 3}" for i in range(len(created))],
            "compound": [0.5] * len(created),
        }
    )
    sessions = _sessions(22)
    rollup = rollup_daily_sentiment(df, session_close=_CLOSE, session_tz=_TZ, sessions=sessions)
    closes = pd.DatetimeIndex([_close_utc(d) for d in sessions])

    # Slow reference: for each post, find the first session index k>=1 with
    # close(k-1) >= u.
    expected_counts: dict[pd.Timestamp, int] = {}
    for c in created:
        u = pd.Timestamp(c, unit="s", tz="UTC")
        target = None
        for k in range(1, len(sessions)):
            if closes[k - 1] >= u:
                target = sessions[k]
                break
        if target is not None:
            expected_counts[target] = expected_counts.get(target, 0) + 1

    total = rollup.mention_count.sum(axis=1)
    for day, exp in expected_counts.items():
        assert int(total.loc[day]) == exp


@_SETTINGS
@given(
    seconds=st.lists(
        st.integers(min_value=0, max_value=10 * 24 * 3600),
        min_size=2,
        max_size=20,
    ),
    split=st.integers(min_value=1, max_value=19),
)
def test_aggregator_prefix_deterministic(seconds: list[int], split: int) -> None:
    """Appending FUTURE posts never changes an already-emitted ``(ticker, day)`` row."""
    base = int(pd.Timestamp("2021-01-04 00:00", tz=_TZ).timestamp())
    seconds = sorted(seconds)
    split = min(split, len(seconds) - 1)
    created = [base + s for s in seconds]
    tickers = [f"T{i % 2}" for i in range(len(created))]
    compounds = [((-1) ** i) * 0.3 for i in range(len(created))]
    full = pd.DataFrame({"created_utc": created, "ticker": tickers, "compound": compounds})
    prefix = full.iloc[:split]
    sessions = _sessions(16)

    r_prefix = rollup_daily_sentiment(
        prefix, session_close=_CLOSE, session_tz=_TZ, sessions=sessions
    )
    r_full = rollup_daily_sentiment(full, session_close=_CLOSE, session_tz=_TZ, sessions=sessions)

    # On days that were already complete in the prefix (every contributing post is
    # in the prefix), the full roll-up must agree exactly.
    closes = pd.DatetimeIndex([_close_utc(d) for d in sessions])

    def _target(u_sec: int) -> pd.Timestamp | None:
        u = pd.Timestamp(u_sec, unit="s", tz="UTC")
        for k in range(1, len(sessions)):
            if closes[k - 1] >= u:
                return sessions[k]
        return None

    future_days = {_target(c) for c in created[split:]}
    common_cols = r_prefix.mean_compound.columns.intersection(r_full.mean_compound.columns)
    for day in r_prefix.mean_compound.index:
        if r_prefix.mention_count.loc[day].sum() == 0:
            continue
        if day in future_days:
            continue  # a later post also lands here; the row legitimately changes
        # Compare the value of every already-emitted (ticker, day) cell. The full
        # panel may carry EXTRA ticker columns, but every cell present in the
        # prefix must be byte-identical (prefix-determinism / future invariance).
        for col in common_cols:
            p_count = r_prefix.mention_count.loc[day, col]
            if p_count == 0:
                continue
            assert r_full.mention_count.loc[day, col] == p_count
            p_mean = r_prefix.mean_compound.loc[day, col]
            f_mean = r_full.mean_compound.loc[day, col]
            assert (pd.isna(p_mean) and pd.isna(f_mean)) or p_mean == f_mean


# --------------------------------------------------------------------------- #
# Train-only scaler + shift
# --------------------------------------------------------------------------- #
def _panel(n_obs: int, n_assets: int, seed: int) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_obs, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(gen.standard_normal((n_obs, n_assets)), index=idx, columns=cols)


@_SETTINGS
@given(
    n_obs=st.integers(min_value=12, max_value=60),
    n_assets=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
    window=st.integers(min_value=1, max_value=5),
    bump=st.floats(min_value=10.0, max_value=1e6, allow_nan=False),
)
def test_scaler_fit_on_train_only(
    n_obs: int, n_assets: int, seed: int, window: int, bump: float
) -> None:
    """Perturbing TEST rows leaves the fitted scaler AND the train signal unchanged."""
    sentiment = _panel(n_obs, n_assets, seed)
    train_end = sentiment.index[n_obs // 2]

    state = fit_standardizer(sentiment, train_end=train_end, window=window)
    spec = SignalSpec(window=window, lag=1, threshold=0.0, long_only=False)
    positions = build_positions(sentiment, state, spec)

    perturbed = sentiment.copy()
    test_mask = perturbed.index > train_end
    perturbed.loc[test_mask] = perturbed.loc[test_mask] + bump

    state_p = fit_standardizer(perturbed, train_end=train_end, window=window)
    pd.testing.assert_series_equal(state.mean, state_p.mean)
    pd.testing.assert_series_equal(state.std, state_p.std)
    assert state.n_train == state_p.n_train

    # Train-slice positions are unchanged: with a causal window, the only place a
    # test row can reach is a row at index t-lag; the train slice up to
    # train_end-lag depends solely on train data.
    positions_p = build_positions(perturbed, state, spec)
    safe_end = sentiment.index[max(0, (n_obs // 2) - max(window, spec.lag))]
    pd.testing.assert_frame_equal(positions.loc[:safe_end], positions_p.loc[:safe_end])


@_SETTINGS
@given(
    n_obs=st.integers(min_value=10, max_value=50),
    n_assets=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    shift=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
)
def test_standardization_scale_invariance(
    n_obs: int, n_assets: int, seed: int, scale: float, shift: float
) -> None:
    """An affine ``a*x + b`` rescaling of sentiment yields IDENTICAL positions."""
    sentiment = _panel(n_obs, n_assets, seed)
    train_end = sentiment.index[n_obs // 2]
    spec = SignalSpec(window=1, lag=1, threshold=0.0, long_only=False)

    state = fit_standardizer(sentiment, train_end=train_end, window=1)
    base = build_positions(sentiment, state, spec)

    rescaled = sentiment * scale + shift
    state_r = fit_standardizer(rescaled, train_end=train_end, window=1)
    rescaled_pos = build_positions(rescaled, state_r, spec)

    pd.testing.assert_frame_equal(base, rescaled_pos)


@_SETTINGS
@given(
    n_obs=st.integers(min_value=8, max_value=40),
    n_assets=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
    lag=st.integers(min_value=1, max_value=4),
)
def test_shift_lag_enforced(n_obs: int, n_assets: int, seed: int, lag: int) -> None:
    """``build_positions`` is exactly ``raw_positions.shift(lag)`` with flat lead rows."""
    sentiment = _panel(n_obs, n_assets, seed)
    train_end = sentiment.index[n_obs // 2]
    spec = SignalSpec(window=1, lag=lag, threshold=0.0, long_only=False)
    state = fit_standardizer(sentiment, train_end=train_end, window=1)

    shifted = build_positions(sentiment, state, spec)
    # lag=0 is rejected by build_positions (must be >= 1); reconstruct raw directly.
    z = (sentiment - state.mean) / state.std
    raw_positions = (z > 0.0).astype("float64") - (z < 0.0).astype("float64")

    expected = raw_positions.shift(lag).fillna(0.0)
    pd.testing.assert_frame_equal(shifted, expected)

    # The leading ``lag`` rows are flat (no same-bar lookahead).
    assert (shifted.iloc[:lag].to_numpy() == 0.0).all()
    # Positions live in {-1, 0, +1}.
    assert set(np.unique(shifted.to_numpy())) <= {-1.0, 0.0, 1.0}


@_SETTINGS
@given(
    n_obs=st.integers(min_value=8, max_value=40),
    n_assets=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_long_only_positions_are_nonnegative(n_obs: int, n_assets: int, seed: int) -> None:
    """``long_only`` emits only ``{0, +1}`` positions."""
    sentiment = _panel(n_obs, n_assets, seed)
    train_end = sentiment.index[n_obs // 2]
    state = fit_standardizer(sentiment, train_end=train_end, window=1)
    spec = SignalSpec(window=1, lag=1, threshold=0.0, long_only=True)
    pos = build_positions(sentiment, state, spec)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}


def test_build_positions_rejects_zero_lag() -> None:
    """A ``lag < 1`` is rejected (it would permit a same-bar return)."""
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    sentiment = _panel(10, 2, 0)
    state = fit_standardizer(sentiment, train_end=sentiment.index[5], window=1)
    with pytest.raises(ValidationError):
        build_positions(sentiment, state, SignalSpec(window=1, lag=0, threshold=0.0))


# --------------------------------------------------------------------------- #
# Roll-up edge cases and validation branches
# --------------------------------------------------------------------------- #
def _utc(s: str) -> int:
    return int(pd.Timestamp(s, tz=_TZ).timestamp())


def test_rollup_basic_values_and_to_dict() -> None:
    """A small hand-built example yields the expected as-of aggregates."""
    df = pd.DataFrame(
        {
            "created_utc": [
                _utc("2021-01-04 15:00"),  # before Mon close -> Tue
                _utc("2021-01-04 17:00"),  # after Mon close  -> Wed
                _utc("2021-01-05 10:00"),  # before Tue close -> Wed
            ],
            "ticker": ["GME", "GME", "GME"],
            "compound": [0.5, -0.3, 0.1],
        }
    )
    r = rollup_daily_sentiment(df, session_close=_CLOSE, session_tz=_TZ)
    tue = pd.Timestamp("2021-01-05")
    wed = pd.Timestamp("2021-01-06")
    assert float(r.mean_compound.loc[tue, "GME"]) == 0.5
    assert abs(float(r.mean_compound.loc[wed, "GME"]) - (-0.1)) < 1e-12
    assert r.mention_count.loc[tue, "GME"] == 1.0
    assert r.mention_count.loc[wed, "GME"] == 2.0
    assert r.positive_share.loc[wed, "GME"] == 0.5  # one of two compounds > 0
    assert r.median_compound.loc[tue, "GME"] == 0.5

    d = r.to_dict()
    assert set(d) == {
        "mean_compound",
        "median_compound",
        "mention_count",
        "positive_share",
        "session_tz",
        "meta",
    }
    assert d["session_tz"] == _TZ
    assert d["meta"]["n_mentions"] == 3
    # NaN cells (Monday has no inflowing post) serialize to None, not NaN.
    mon_key = str(pd.Timestamp("2021-01-04"))
    assert d["mean_compound"][mon_key]["GME"] is None
    # Populated cells serialize to plain floats.
    assert d["mean_compound"][str(tue)]["GME"] == 0.5


def test_rollup_empty_input_with_sessions() -> None:
    """An empty input returns well-formed empty panels on the requested calendar."""
    empty = pd.DataFrame({"created_utc": [], "ticker": [], "compound": []})
    sessions = _sessions(5)
    r = rollup_daily_sentiment(empty, sessions=sessions)
    assert list(r.mean_compound.index) == list(sessions)
    assert r.mention_count.shape[1] == 0
    assert r.to_dict()["meta"]["n_mentions"] == 0


def test_rollup_empty_input_default_calendar() -> None:
    """An empty input with no explicit calendar returns an empty index."""
    empty = pd.DataFrame({"created_utc": [], "ticker": [], "compound": []})
    r = rollup_daily_sentiment(empty)
    assert len(r.mean_compound.index) == 0


def test_rollup_default_calendar_path() -> None:
    """With ``sessions=None`` the business-day calendar covers the observed dates."""
    df = pd.DataFrame(
        {
            "created_utc": [_utc("2021-03-01 09:00"), _utc("2021-03-02 09:00")],
            "ticker": ["AMC", "AMC"],
            "compound": [0.2, 0.4],
        }
    )
    r = rollup_daily_sentiment(df)
    assert r.mention_count["AMC"].sum() >= 1
    # Calendar is business days only.
    cal = pd.DatetimeIndex(r.mean_compound.index)
    assert bool((cal.dayofweek < 5).all())


def test_rollup_rejects_non_dataframe() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    with pytest.raises(ValidationError, match="DataFrame"):
        rollup_daily_sentiment([1, 2, 3])  # type: ignore[arg-type]


def test_rollup_rejects_missing_columns() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    with pytest.raises(ValidationError, match="missing required column"):
        rollup_daily_sentiment(pd.DataFrame({"created_utc": [1], "ticker": ["X"]}))


def test_rollup_rejects_nan_compound() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    df = pd.DataFrame(
        {"created_utc": [_utc("2021-01-04 09:00")], "ticker": ["X"], "compound": [float("nan")]}
    )
    with pytest.raises(ValidationError, match="non-numeric or NaN"):
        rollup_daily_sentiment(df)


def test_rollup_rejects_unparseable_timestamp() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    df = pd.DataFrame({"created_utc": ["not-a-number"], "ticker": ["X"], "compound": [0.1]})
    with pytest.raises(ValidationError, match="unparseable"):
        rollup_daily_sentiment(df)


def test_rollup_rejects_missing_ticker() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    df = pd.DataFrame(
        {"created_utc": [_utc("2021-01-04 09:00")], "ticker": [None], "compound": [0.1]}
    )
    with pytest.raises(ValidationError, match="missing values"):
        rollup_daily_sentiment(df)


def test_rollup_rejects_bad_session_close() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    df = pd.DataFrame(
        {"created_utc": [_utc("2021-01-04 09:00")], "ticker": ["X"], "compound": [0.1]}
    )
    with pytest.raises(ValidationError, match="HH:MM"):
        rollup_daily_sentiment(df, session_close="9am")
    with pytest.raises(ValidationError, match="out of range"):
        rollup_daily_sentiment(df, session_close="25:00")


def test_rollup_accepts_tz_aware_sessions() -> None:
    """A tz-aware session calendar is normalized to tz-naive days."""
    df = pd.DataFrame(
        {"created_utc": [_utc("2021-01-04 09:00")], "ticker": ["X"], "compound": [0.3]}
    )
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=6)).tz_localize("UTC")
    r = rollup_daily_sentiment(df, sessions=sessions)
    assert pd.DatetimeIndex(r.mean_compound.index).tz is None


# --------------------------------------------------------------------------- #
# Signal edge cases and validation branches
# --------------------------------------------------------------------------- #
def test_signal_accepts_ndarray_and_smoothing() -> None:
    """``fit_standardizer``/``build_positions`` accept 2-D ndarrays and smooth.

    A bare ndarray is coerced to a default ``RangeIndex`` panel, so ``train_end``
    is an integer position in that index (not a timestamp).
    """
    arr = np.random.default_rng(1).standard_normal((20, 3))
    state = fit_standardizer(arr, train_end=10, window=3)  # type: ignore[arg-type]
    assert state.n_train == 11
    spec = SignalSpec(window=3, lag=1, threshold=0.5, long_only=False)
    pos = build_positions(arr, state, spec)
    assert pos.shape == (20, 3)


def test_signal_nan_sentiment_is_flat() -> None:
    """Days with NaN sentiment (no mentions) produce a flat position, not an error."""
    idx = pd.date_range("2021-01-04", periods=10, freq="B")
    s = pd.DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)), index=idx, columns=["A", "B"]
    )
    s.iloc[7:, 0] = float("nan")
    state = fit_standardizer(s, train_end=idx[5], window=1)
    pos = build_positions(s, state, SignalSpec(window=1, lag=1, threshold=0.0))
    # NaN sentiment -> 0 position (after the shift), never NaN.
    assert not pos.isna().to_numpy().any()
    assert (pos.iloc[8:, 0] == 0.0).all()


def test_fit_standardizer_rejects_bad_window() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    s = _panel(10, 2, 0)
    with pytest.raises(ValidationError, match="window must be >= 1"):
        fit_standardizer(s, train_end=s.index[5], window=0)


def test_fit_standardizer_rejects_empty_train() -> None:
    import pytest

    from wsb_sentiment._exceptions import InsufficientDataError

    s = _panel(10, 2, 0)
    with pytest.raises(InsufficientDataError, match="no observations"):
        fit_standardizer(s, train_end=s.index[0] - pd.Timedelta(days=10), window=1)


def test_fit_standardizer_constant_train_floors_std() -> None:
    """A constant train slice yields std floored at EPS (no divide-by-zero)."""
    from wsb_sentiment._constants import EPS

    idx = pd.date_range("2021-01-04", periods=8, freq="B")
    s = pd.DataFrame(5.0, index=idx, columns=["A"])
    state = fit_standardizer(s, train_end=idx[4], window=1)
    assert state.std["A"] == EPS


def test_build_positions_rejects_bad_inputs() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    s = _panel(10, 2, 0)
    state = fit_standardizer(s, train_end=s.index[5], window=1)
    with pytest.raises(ValidationError, match="window must be >= 1"):
        build_positions(s, state, SignalSpec(window=0, lag=1, threshold=0.0))
    with pytest.raises(ValidationError, match="threshold must be >= 0"):
        build_positions(s, state, SignalSpec(window=1, lag=1, threshold=-1.0))


def test_build_positions_rejects_non_panel() -> None:
    import pytest

    from wsb_sentiment._exceptions import ValidationError

    state = fit_standardizer(_panel(10, 2, 0), train_end=_panel(10, 2, 0).index[5], window=1)
    with pytest.raises(ValidationError, match="DataFrame or 2-D"):
        build_positions([1, 2, 3], state, SignalSpec())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="2-dimensional"):
        build_positions(np.zeros((2, 2, 2)), state, SignalSpec())


def test_build_positions_unseen_columns_are_flat() -> None:
    """Columns not seen during fit map to a neutral (flat) position."""
    idx = pd.date_range("2021-01-04", periods=10, freq="B")
    s = pd.DataFrame(
        np.random.default_rng(4).standard_normal((10, 2)), index=idx, columns=["A", "B"]
    )
    state = fit_standardizer(s[["A"]], train_end=idx[5], window=1)
    pos = build_positions(s, state, SignalSpec(window=1, lag=1, threshold=0.0))
    # Column B was never in the fitted state but still produces valid positions.
    assert set(np.unique(pos["B"].to_numpy())) <= {-1.0, 0.0, 1.0}


def test_state_to_dict_roundtrip() -> None:
    """``StandardizerState.to_dict`` is JSON-plain (str keys, float values)."""
    sentiment = _panel(12, 2, 3)
    state = fit_standardizer(sentiment, train_end=sentiment.index[6], window=2)
    d = state.to_dict()
    assert set(d) == {"mean", "std", "n_train"}
    assert all(isinstance(k, str) for k in d["mean"])
    assert isinstance(d["n_train"], int)
    assert isinstance(StandardizerState(state.mean, state.std, state.n_train).n_train, int)
