"""Unit tests for the vendored reuse primitives exercised by the honest-stats layer.

Covers the public reuse surface copied from the HRP / pairs-trading repos that the
verdict pipeline depends on but that the higher-level pipeline tests do not drive
end to end: the purge/embargo helpers, the Memmel-JK Sharpe-equality test, the
stationary-bootstrap CI, the immutable result containers, the walk-forward driver,
the scalar performance statistics, and the seeded RNG / validation / manifest
infrastructure. Each is tested as the public API it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._exceptions import (
    InsufficientDataError,
    ValidationError,
)
from wsb_sentiment._manifest import RunManifest, config_hash
from wsb_sentiment._rng import make_rng, spawn_substreams
from wsb_sentiment._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from wsb_sentiment.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from wsb_sentiment.backtest.walk_forward import walk_forward_backtest
from wsb_sentiment.evaluation._purge import embargo_indices, purge_indices
from wsb_sentiment.evaluation.bootstrap_ci import stationary_bootstrap_ci
from wsb_sentiment.evaluation.cpcv import cpcv_paths
from wsb_sentiment.evaluation.memmel import memmel_test
from wsb_sentiment.evaluation.results import (
    BootstrapCI,
    CPCVResult,
    DSRResult,
    MemmelResult,
    PBOResult,
)

# --------------------------------------------------------------------------- #
# Purge / embargo helpers                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_purge_indices_drops_overlapping_label_windows() -> None:
    """Train dates whose label horizon touches the test span are purged."""
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    train = idx[:20]
    test = idx[25:35]
    kept = purge_indices(train, test, label_horizon_days=3)
    # Every surviving train date's label window ends strictly before the test start.
    assert (kept + pd.Timedelta(days=3) < test.min()).all()
    # Zero horizon keeps everything before the test span.
    assert len(purge_indices(train, test, label_horizon_days=0)) == len(train)


@pytest.mark.unit
def test_purge_indices_empty_and_validation() -> None:
    """Empty inputs are returned unchanged; bad types/horizons raise."""
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    empty = idx[:0]
    assert len(purge_indices(idx, empty, label_horizon_days=2)) == len(idx)
    with pytest.raises(ValidationError, match="non-negative"):
        purge_indices(idx, idx, label_horizon_days=-1)
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        purge_indices(pd.Index([1, 2, 3]), idx, label_horizon_days=1)


@pytest.mark.unit
def test_embargo_indices_window() -> None:
    """Embargo returns exactly the train dates inside the post-test window."""
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    test = idx[10:20]
    train = idx
    dropped = embargo_indices(train, test, embargo_days=5)
    assert (dropped > test.max()).all()
    assert (dropped <= test.max() + pd.Timedelta(days=5)).all()
    # Zero embargo drops nothing.
    assert len(embargo_indices(train, test, embargo_days=0)) == 0
    with pytest.raises(ValidationError, match="non-negative"):
        embargo_indices(train, test, embargo_days=-2)


# --------------------------------------------------------------------------- #
# Memmel-JK Sharpe-equality test                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_memmel_test_identical_streams_has_no_difference() -> None:
    """Two identical streams have equal Sharpe and a non-significant z-stat."""
    gen = make_rng(101)
    a = pd.Series(gen.normal(0.001, 0.01, size=300))
    result = memmel_test(a, a.copy())
    assert isinstance(result, MemmelResult)
    assert result.sr_a == pytest.approx(result.sr_b)
    assert result.z_stat == pytest.approx(0.0, abs=1e-9)
    assert result.correlation == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= result.p_value <= 1.0


@pytest.mark.unit
def test_memmel_test_validation() -> None:
    """Length mismatch, too-few observations, and flat streams raise."""
    gen = make_rng(102)
    a = gen.normal(size=20)
    with pytest.raises(ValidationError, match="same length"):
        memmel_test(a, gen.normal(size=19))
    with pytest.raises(ValidationError, match="four paired"):
        memmel_test(a[:3], gen.normal(size=3))
    with pytest.raises(ValidationError, match="positive"):
        memmel_test(np.ones(10), gen.normal(size=10))


# --------------------------------------------------------------------------- #
# Stationary bootstrap CI                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stationary_bootstrap_ci_brackets_the_mean() -> None:
    """The bootstrap CI brackets the point estimate of the statistic."""
    gen = make_rng(7)
    data = pd.Series(gen.normal(0.05, 1.0, size=400))
    result = stationary_bootstrap_ci(data, lambda x: float(np.mean(x)), n_boot=300, rng=make_rng(7))
    assert isinstance(result, BootstrapCI)
    assert result.ci_low <= result.point_estimate <= result.ci_high
    assert result.expected_block >= 2


@pytest.mark.unit
def test_stationary_bootstrap_ci_validation() -> None:
    """Bad alpha / n_boot / too-short series raise ValidationError."""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValidationError, match="alpha"):
        stationary_bootstrap_ci(data, lambda x: float(x.mean()), alpha=1.5)
    with pytest.raises(ValidationError, match="n_boot"):
        stationary_bootstrap_ci(data, lambda x: float(x.mean()), n_boot=0)
    with pytest.raises(ValidationError, match="two finite"):
        stationary_bootstrap_ci(np.array([np.nan, 1.0]), lambda x: float(x.mean()))


# --------------------------------------------------------------------------- #
# Result containers                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_result_containers_roundtrip_and_validate() -> None:
    """The frozen result dataclasses serialize and enforce their invariants."""
    dsr = DSRResult(
        realized_sr=0.1,
        deflated_threshold=0.05,
        psr_of_threshold=0.6,
        dsr=0.4,
        p_value=0.2,
        n_trials_effective=3.0,
        sample_size=252,
    )
    assert dsr.to_dict()["sample_size"] == 252

    pbo = PBOResult(pbo=0.42, logit_lambdas=(0.1, -0.2), n_splits=8, s_partitions=8)
    assert 0.0 <= pbo.to_dict()["pbo"] <= 1.0

    memmel = MemmelResult(sr_a=0.1, sr_b=0.2, z_stat=-1.0, p_value=0.3, n_obs=100, correlation=0.5)
    assert memmel.to_dict()["n_obs"] == 100

    boot = BootstrapCI(
        point_estimate=0.1, ci_low=0.0, ci_high=0.2, alpha=0.05, n_boot=500, expected_block=4
    )
    assert boot.to_dict()["expected_block"] == 4

    cpcv = CPCVResult(
        paths=(pd.Series([0.1, 0.2]),),
        n_groups=5,
        k_test=2,
        n_combinations=10,
        path_sharpes=(0.3,),
        median_path_sharpe=0.3,
    )
    assert cpcv.to_dict()["n_groups"] == 5

    with pytest.raises(ValueError, match="dsr must lie"):
        DSRResult(
            realized_sr=0.1,
            deflated_threshold=0.0,
            psr_of_threshold=0.5,
            dsr=1.5,
            p_value=0.1,
            n_trials_effective=2.0,
            sample_size=10,
        )
    with pytest.raises(ValueError, match="s_partitions"):
        PBOResult(pbo=0.1, logit_lambdas=(), n_splits=4, s_partitions=7)
    with pytest.raises(ValueError, match="ci_low"):
        BootstrapCI(
            point_estimate=0.1, ci_low=0.5, ci_high=0.2, alpha=0.05, n_boot=10, expected_block=2
        )


# --------------------------------------------------------------------------- #
# Walk-forward driver + scalar statistics                                      #
# --------------------------------------------------------------------------- #


def _equal_weight(window: pd.DataFrame) -> pd.Series:
    """A trivial allocator: equal weight across the in-sample columns."""
    n = window.shape[1]
    return pd.Series(1.0 / n, index=window.columns, dtype="float64")


@pytest.mark.unit
def test_walk_forward_backtest_runs_and_is_cost_monotone() -> None:
    """The walk-forward driver produces aligned OOS series; net Sharpe falls with cost."""
    gen = make_rng(5)
    index = pd.date_range("2021-01-01", periods=400, freq="B")
    panel = pd.DataFrame(
        gen.normal(0.0005, 0.01, size=(400, 3)),
        index=index,
        columns=["A", "B", "C"],
    )
    cheap = walk_forward_backtest(
        panel, _equal_weight, lookback_window=60, rebalance="monthly", cost_bps=1.0
    )
    dear = walk_forward_backtest(
        panel, _equal_weight, lookback_window=60, rebalance="monthly", cost_bps=50.0
    )
    assert cheap.n_rebalances > 0
    assert len(cheap.oos_returns) == len(cheap.gross_returns)
    # Higher per-side cost cannot raise the net mean OOS return.
    assert dear.oos_returns.mean() <= cheap.oos_returns.mean() + 1e-12
    # The result serializes cleanly across the API boundary.
    payload = cheap.to_dict()
    assert payload["n_rebalances"] == cheap.n_rebalances
    assert set(payload) >= {"oos_returns", "weights", "turnover", "costs"}


@pytest.mark.unit
def test_walk_forward_backtest_validation() -> None:
    """Bad cost / rebalance / lookback and too-short panels raise."""
    index = pd.date_range("2021-01-01", periods=80, freq="B")
    panel = pd.DataFrame(np.zeros((80, 2)), index=index, columns=["A", "B"])
    with pytest.raises(ValidationError, match="cost_bps"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=20, cost_bps=-1.0)
    with pytest.raises(ValidationError, match="rebalance"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=20, rebalance="weekly")
    with pytest.raises(ValidationError, match="lookback_window"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=2)
    with pytest.raises(InsufficientDataError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=79)


@pytest.mark.unit
def test_scalar_performance_statistics() -> None:
    """Sharpe, vol, turnover and max-drawdown behave sensibly on known inputs."""
    rets = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005])
    assert np.isfinite(sharpe_ratio(rets))
    assert annualized_vol(rets) >= 0.0
    # A flat series has undefined (NaN) Sharpe and zero volatility.
    flat = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(flat))
    assert annualized_vol(flat) == pytest.approx(0.0)
    # Turnover between disjoint weight books is one-way 1.0.
    prev = pd.Series({"A": 1.0, "B": 0.0})
    new = pd.Series({"A": 0.0, "B": 1.0})
    assert turnover(prev, new) == pytest.approx(1.0)
    # A monotonically rising equity has zero drawdown; a dip is negative.
    assert max_drawdown(pd.Series([0.01, 0.01, 0.01])) == pytest.approx(0.0)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.0])) < 0.0


# --------------------------------------------------------------------------- #
# Combinatorial purged cross-validation                                        #
# --------------------------------------------------------------------------- #


def _first_two_columns(prices: pd.DataFrame) -> tuple[str, str]:
    """A trivial 'pair selector': always pick the first two price columns."""
    cols = list(prices.columns)
    return str(cols[0]), str(cols[1])


def _spread_returns(prices: pd.DataFrame, selection: tuple[str, str]) -> pd.Series:
    """A trivial 'pair backtester': the daily spread return of the chosen pair."""
    a, b = selection
    spread = prices[a].pct_change(fill_method=None) - prices[b].pct_change(fill_method=None)
    return spread.dropna()


@pytest.mark.unit
def test_cpcv_paths_reassembles_synthetic_paths() -> None:
    """CPCV runs over a price panel and reassembles the expected number of paths."""
    gen = make_rng(3)
    index = pd.date_range("2021-01-01", periods=120, freq="B")
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(gen.normal(0.0, 0.01, size=(120, 3)), axis=0)),
        index=index,
        columns=["X", "Y", "Z"],
    )
    result = cpcv_paths(
        prices,
        n_groups=6,
        k_test=2,
        purge_days=2,
        embargo_pct=0.0,
        pair_selector=_first_two_columns,
        pair_backtester=_spread_returns,
    )
    assert isinstance(result, CPCVResult)
    # Path count formula C(N, k) * k / N = C(6, 2) * 2 / 6 = 5.
    assert result.n_combinations == 15
    assert len(result.path_sharpes) == len(result.paths)


@pytest.mark.unit
def test_cpcv_paths_validation() -> None:
    """CPCV rejects non-datetime indexes and bad group/test-group counts."""
    bad = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    with pytest.raises(ValidationError, match="DatetimeIndex"):
        cpcv_paths(
            bad,
            pair_selector=_first_two_columns,
            pair_backtester=_spread_returns,
        )
    index = pd.date_range("2021-01-01", periods=40, freq="B")
    prices = pd.DataFrame(np.ones((40, 2)), index=index, columns=["A", "B"])
    with pytest.raises(ValidationError, match="n_groups"):
        cpcv_paths(
            prices,
            n_groups=1,
            pair_selector=_first_two_columns,
            pair_backtester=_spread_returns,
        )
    with pytest.raises(ValidationError, match="k_test"):
        cpcv_paths(
            prices,
            n_groups=4,
            k_test=4,
            pair_selector=_first_two_columns,
            pair_backtester=_spread_returns,
        )


# --------------------------------------------------------------------------- #
# RNG / validation / manifest infrastructure                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_rng_reproducible_and_substreams_independent() -> None:
    """Seeded generators reproduce; substreams are independent and deterministic."""
    a = make_rng(42).standard_normal(8)
    b = make_rng(42).standard_normal(8)
    assert np.array_equal(a, b)
    kids1 = spawn_substreams(9, 3)
    kids2 = spawn_substreams(9, 3)
    assert len(kids1) == 3
    draws1 = [k.standard_normal(4) for k in kids1]
    draws2 = [k.standard_normal(4) for k in kids2]
    for d1, d2 in zip(draws1, draws2, strict=True):
        assert np.array_equal(d1, d2)
    # Distinct substreams are not identical.
    assert not np.array_equal(draws1[0], draws1[1])
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(1, -1)


@pytest.mark.unit
def test_validation_helpers() -> None:
    """The coercion helpers enforce shape/NaN and align on the common index."""
    assert ensure_series([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValidationError, match="NaN"):
        ensure_series([1.0, np.nan])
    with pytest.raises(ValidationError, match="1-dimensional"):
        ensure_series(np.zeros((2, 2)))
    df = ensure_dataframe(np.ones((3, 2)), columns=["x", "y"])
    assert list(df.columns) == ["x", "y"]
    with pytest.raises(ValidationError, match="2-dimensional"):
        ensure_dataframe(np.ones((2, 2, 2)))
    with pytest.raises(ValidationError, match="NaN"):
        ensure_dataframe(np.array([[1.0, np.nan]]))

    left = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2021-01-01", periods=3))
    right = pd.DataFrame({"b": [4, 5]}, index=pd.date_range("2021-01-02", periods=2))
    la, ra = align_inner(left, right)
    assert list(la.index) == list(ra.index)
    assert len(la) == 2
    with pytest.raises(ValidationError, match="common index"):
        align_inner(left, right.set_index(pd.date_range("2030-01-01", periods=2)))

    validate_min_obs(left, 3)
    with pytest.raises(InsufficientDataError):
        validate_min_obs(left, 10)


@pytest.mark.unit
def test_manifest_config_hash_and_capture() -> None:
    """Config hashing is order-independent; capture builds a serializable manifest."""
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2 and len(h1) == 32
    assert config_hash({"a": 1}) != h1

    manifest = RunManifest.capture({"window": 1, "lag": 1}, seed=7)
    payload = manifest.to_dict()
    assert payload["seed"] == 7
    assert isinstance(payload["git_sha"], str)
    assert isinstance(payload["dirty"], bool)
    assert len(payload["config_hash"]) == 32
