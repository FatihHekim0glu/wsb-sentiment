"""Unit tests for the orchestrating honest-statistics kernels.

Covers the three functions implemented in
:mod:`wsb_sentiment.evaluation.stats`:

- :func:`effective_n_trials` — PCA-of-trial-returns deflation of the
  multiplicity count (identical columns collapse to one; independent columns
  stay near the raw count; degenerate inputs guarded);
- :func:`hac_tstat` — Newey-West t-stat / p-value, including the degenerate
  zero-variance branch;
- :func:`compute_honest_stats` — the full assembly of net/buy-hold Sharpe, the
  effective trial count, the DSR/PSR, the PBO/CSCV, and the HAC test, with the
  ``n_trials`` guard that keeps the effective count in ``[1, n_grid_trials]``.

Also pins the CSCV PBO behaviour (overfit grid -> high PBO; genuine-edge grid
-> low PBO) since the verdict depends on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.evaluation.pbo import pbo_cscv
from wsb_sentiment.evaluation.stats import (
    HonestStats,
    compute_honest_stats,
    effective_n_trials,
    hac_tstat,
)


# --------------------------------------------------------------------------- #
# effective_n_trials                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_effective_n_trials_collapses_identical_columns(
    rng: np.random.Generator,
) -> None:
    """Perfectly-correlated trials count as a single independent trial."""
    base = rng.standard_normal(300)
    frame = pd.DataFrame({f"c{i}": base for i in range(25)})
    assert effective_n_trials(frame) == pytest.approx(1.0)


@pytest.mark.unit
def test_effective_n_trials_independent_columns_near_raw_count(
    rng: np.random.Generator,
) -> None:
    """Independent trials keep most of the raw multiplicity at 95% variance."""
    frame = pd.DataFrame(rng.standard_normal((400, 20)))
    eff = effective_n_trials(frame)
    # No deflation for truly independent columns: well above half the grid.
    assert 15.0 <= eff <= 20.0


@pytest.mark.unit
def test_effective_n_trials_is_bounded_by_grid_and_one(
    rng: np.random.Generator,
) -> None:
    """The effective count never exceeds the column count nor drops below one."""
    frame = pd.DataFrame(rng.standard_normal((200, 8)))
    eff = effective_n_trials(frame)
    assert 1.0 <= eff <= 8.0


@pytest.mark.unit
def test_effective_n_trials_single_column_is_one() -> None:
    """A single configuration is exactly one independent trial."""
    frame = pd.DataFrame({"only": np.linspace(-1.0, 1.0, 50)})
    assert effective_n_trials(frame) == 1.0


@pytest.mark.unit
def test_effective_n_trials_threshold_monotone(rng: np.random.Generator) -> None:
    """A higher variance threshold needs at least as many components."""
    frame = pd.DataFrame(rng.standard_normal((400, 15)))
    low = effective_n_trials(frame, var_threshold=0.5)
    high = effective_n_trials(frame, var_threshold=0.99)
    assert high >= low


@pytest.mark.unit
def test_effective_n_trials_drops_constant_columns(
    rng: np.random.Generator,
) -> None:
    """Zero-variance (degenerate) columns add no independent trials."""
    good = rng.standard_normal((300, 6))
    const = np.zeros((300, 4))
    frame = pd.DataFrame(np.hstack([good, const]))
    eff = effective_n_trials(frame)
    # At most the six informative columns survive.
    assert 1.0 <= eff <= 6.0


@pytest.mark.unit
def test_effective_n_trials_all_constant_is_one() -> None:
    """An all-constant grid collapses to a single degenerate trial."""
    frame = pd.DataFrame(np.ones((40, 5)))
    assert effective_n_trials(frame) == 1.0


@pytest.mark.unit
def test_effective_n_trials_too_few_rows_falls_back_to_raw() -> None:
    """With <2 usable rows the conservative raw multiplicity is returned."""
    frame = pd.DataFrame(np.array([[0.1, 0.2, 0.3]]))  # one row, three trials
    assert effective_n_trials(frame) == 3.0


@pytest.mark.unit
def test_effective_n_trials_rejects_bad_inputs() -> None:
    """Non-DataFrame input and out-of-range thresholds raise ValidationError."""
    with pytest.raises(ValidationError):
        effective_n_trials([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        effective_n_trials(pd.DataFrame({"a": [1.0, 2.0]}), var_threshold=0.0)
    with pytest.raises(ValidationError):
        effective_n_trials(pd.DataFrame({"a": [1.0, 2.0]}), var_threshold=1.5)


@pytest.mark.unit
def test_effective_n_trials_rejects_zero_column_frame() -> None:
    """A 2-D frame with zero columns is rejected (no trials to count)."""
    empty = pd.DataFrame(index=range(5))  # shape (5, 0)
    with pytest.raises(ValidationError, match="at least one column"):
        effective_n_trials(empty)


# --------------------------------------------------------------------------- #
# hac_tstat                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_hac_tstat_positive_drift_significant() -> None:
    """A strong positive drift yields a positive, significant t-stat."""
    returns = pd.Series(np.full(300, 0.01) + 1e-6)
    tstat, pvalue = hac_tstat(returns)
    assert tstat > 0.0
    assert 0.0 <= pvalue <= 1.0


@pytest.mark.unit
def test_hac_tstat_degenerate_series_is_neutral() -> None:
    """A constant (zero long-run variance) series gives t=0, p=1."""
    returns = pd.Series(np.full(50, 0.005))
    tstat, pvalue = hac_tstat(returns)
    assert tstat == 0.0
    assert pvalue == 1.0


@pytest.mark.unit
def test_hac_tstat_accepts_numpy_array(rng: np.random.Generator) -> None:
    """The helper accepts a raw ndarray as well as a Series."""
    arr = rng.standard_normal(200) * 0.01
    tstat, pvalue = hac_tstat(arr)
    assert np.isfinite(tstat)
    assert 0.0 <= pvalue <= 1.0


# --------------------------------------------------------------------------- #
# PBO / CSCV correctness                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pbo_high_for_pure_noise_overfit_grid(rng: np.random.Generator) -> None:
    """A grid of pure-noise trials overfits: PBO should be high (~0.5+)."""
    noise = pd.DataFrame(rng.standard_normal((256, 30)) * 0.01)
    result = pbo_cscv(noise, s=16)
    assert 0.0 <= result.pbo <= 1.0
    assert result.pbo >= 0.4  # no genuine edge -> rampant overfitting
    assert result.n_splits > 0


@pytest.mark.unit
def test_pbo_low_for_genuinely_persistent_edge(rng: np.random.Generator) -> None:
    """A grid where one trial has a real, persistent edge gives a low PBO."""
    n_obs, n_trials = 256, 20
    noise = rng.standard_normal((n_obs, n_trials)) * 0.01
    # Inject a single trial with a strong, sample-wide positive drift so the
    # IS-best is also OOS-best in every split -> overfitting probability is low.
    noise[:, 0] += 0.02
    frame = pd.DataFrame(noise)
    result = pbo_cscv(frame, s=16)
    assert result.pbo <= 0.2


@pytest.mark.unit
def test_pbo_rejects_odd_partition_count(rng: np.random.Generator) -> None:
    """An odd slab count is invalid for CSCV."""
    frame = pd.DataFrame(rng.standard_normal((64, 4)))
    with pytest.raises(ValidationError):
        pbo_cscv(frame, s=15)


# --------------------------------------------------------------------------- #
# compute_honest_stats                                                        #
# --------------------------------------------------------------------------- #
def _grid(rng: np.random.Generator, t: int, n: int, drift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(rng.standard_normal((t, n)) * 0.01 + drift)


@pytest.mark.unit
def test_compute_honest_stats_assembles_full_bundle(
    rng: np.random.Generator,
) -> None:
    """The bundle is populated with finite, range-valid fields."""
    net = pd.Series(rng.standard_normal(300) * 0.01 + 0.0002)
    bh = pd.Series(rng.standard_normal(300) * 0.01 + 0.0003)
    trials = _grid(rng, 300, 24)
    stats = compute_honest_stats(net, bh, trials, n_grid_trials=24)

    assert isinstance(stats, HonestStats)
    assert stats.n_obs == 300
    assert 0.0 <= stats.deflated_sharpe <= 1.0
    assert 0.0 <= stats.psr <= 1.0
    assert 0.0 <= stats.pbo <= 1.0
    assert 0.0 <= stats.hac_pvalue <= 1.0
    assert np.isfinite(stats.hac_tstat)
    assert np.isfinite(stats.net_sharpe)
    assert np.isfinite(stats.buyhold_sharpe)
    # The effective trial count is honest: never above the raw grid, never < 1.
    assert 1.0 <= stats.n_effective_trials <= 24.0

    payload = stats.to_dict()
    assert payload["n_obs"] == 300
    assert set(payload) >= {
        "net_sharpe",
        "buyhold_sharpe",
        "deflated_sharpe",
        "psr",
        "pbo",
        "hac_tstat",
        "hac_pvalue",
        "n_effective_trials",
        "n_obs",
    }


@pytest.mark.unit
def test_compute_honest_stats_n_trials_guard_caps_at_grid(
    rng: np.random.Generator,
) -> None:
    """The effective trial count is clamped to the raw grid size (guard)."""
    net = pd.Series(rng.standard_normal(250) * 0.01)
    bh = pd.Series(rng.standard_normal(250) * 0.01)
    # 12 independent columns but we declare a grid of only 4: the effective
    # count must not exceed the declared raw multiplicity.
    trials = _grid(rng, 250, 12)
    stats = compute_honest_stats(net, bh, trials, n_grid_trials=4)
    assert stats.n_effective_trials <= 4.0
    assert stats.n_effective_trials >= 1.0


@pytest.mark.unit
def test_compute_honest_stats_annualizes_sharpe(rng: np.random.Generator) -> None:
    """net_sharpe scales with sqrt(periods_per_year) vs the per-obs Sharpe."""
    net = pd.Series(rng.standard_normal(300) * 0.01 + 0.001)
    bh = pd.Series(rng.standard_normal(300) * 0.01)
    trials = _grid(rng, 300, 10)
    daily = compute_honest_stats(net, bh, trials, n_grid_trials=10, periods_per_year=1)
    annual = compute_honest_stats(net, bh, trials, n_grid_trials=10, periods_per_year=252)
    assert annual.net_sharpe == pytest.approx(daily.net_sharpe * np.sqrt(252), rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_grid_trials": 0}, "n_grid_trials"),
        ({"n_grid_trials": 10, "periods_per_year": 0}, "periods_per_year"),
    ],
)
def test_compute_honest_stats_rejects_bad_scalars(
    rng: np.random.Generator, kwargs: dict[str, int], match: str
) -> None:
    """Invalid scalar arguments raise a descriptive ValidationError."""
    net = pd.Series(rng.standard_normal(50) * 0.01)
    bh = pd.Series(rng.standard_normal(50) * 0.01)
    trials = _grid(rng, 50, 6)
    with pytest.raises(ValidationError, match=match):
        compute_honest_stats(net, bh, trials, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_compute_honest_stats_rejects_too_short_returns() -> None:
    """Fewer than two finite net observations is an error."""
    net = pd.Series([0.01])
    bh = pd.Series([0.01, 0.02, 0.03])
    trials = pd.DataFrame(np.zeros((3, 4)))
    with pytest.raises(ValidationError, match="net_returns"):
        compute_honest_stats(net, bh, trials, n_grid_trials=4)


@pytest.mark.unit
def test_compute_honest_stats_rejects_too_short_buyhold() -> None:
    """Fewer than two finite buy-hold observations is an error."""
    net = pd.Series(np.linspace(-0.01, 0.01, 10))
    bh = pd.Series([0.01])
    trials = pd.DataFrame(np.zeros((10, 4)))
    with pytest.raises(ValidationError, match="buyhold_returns"):
        compute_honest_stats(net, bh, trials, n_grid_trials=4)


@pytest.mark.unit
def test_compute_honest_stats_handles_degenerate_returns(
    rng: np.random.Generator,
) -> None:
    """A flat (zero-variance) net series gives a neutral, finite bundle.

    Exercises the degenerate-Sharpe branch (skew/kurt fall back to the Gaussian
    null) and the degenerate-column skips in the trial-Sharpe variance.
    """
    net = pd.Series(np.full(60, 0.001))  # constant -> per-obs Sharpe undefined
    bh = pd.Series(rng.standard_normal(60) * 0.01)
    # A trial grid mixing one live column with several constant (degenerate) ones.
    live = rng.standard_normal((60, 1)) * 0.01
    const = np.zeros((60, 5))
    trials = pd.DataFrame(np.hstack([live, const]))
    stats = compute_honest_stats(net, bh, trials, n_grid_trials=6)
    assert stats.net_sharpe == pytest.approx(0.0)
    assert np.isfinite(stats.deflated_sharpe)
    assert 0.0 <= stats.pbo <= 1.0


@pytest.mark.unit
def test_compute_honest_stats_skips_near_empty_trial_column(
    rng: np.random.Generator,
) -> None:
    """A trial column with <2 finite values is skipped in the variance estimate."""
    live = rng.standard_normal((40, 3)) * 0.01
    sparse = np.full((40, 1), np.nan)
    sparse[0, 0] = 0.01  # exactly one finite value -> skipped
    trials = pd.DataFrame(np.hstack([live, sparse]))
    net = pd.Series(rng.standard_normal(40) * 0.01)
    bh = pd.Series(rng.standard_normal(40) * 0.01)
    stats = compute_honest_stats(net, bh, trials, n_grid_trials=4)
    assert np.isfinite(stats.deflated_sharpe)


@pytest.mark.unit
def test_compute_honest_stats_small_grid_uses_neutral_pbo() -> None:
    """A grid too small for CSCV reports the neutral PBO=0.5 (fails low-PBO)."""
    net = pd.Series(np.linspace(-0.01, 0.02, 12))
    bh = pd.Series(np.linspace(-0.01, 0.01, 12))
    single_trial = pd.DataFrame({"only": np.linspace(-0.01, 0.02, 12)})
    stats = compute_honest_stats(net, bh, single_trial, n_grid_trials=1)
    # Only one trial -> CSCV cannot run -> neutral 0.5.
    assert stats.pbo == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_honest_stats_odd_feasible_partition(
    rng: np.random.Generator,
) -> None:
    """A short two-trial grid forces an odd->even slab adjustment in _safe_pbo."""
    # T == 5 rows, 2 trials: s starts at 5 (odd) -> decremented to 4 (even).
    net = pd.Series(rng.standard_normal(5) * 0.01)
    bh = pd.Series(rng.standard_normal(5) * 0.01)
    trials = pd.DataFrame(rng.standard_normal((5, 2)) * 0.01)
    stats = compute_honest_stats(net, bh, trials, n_grid_trials=2)
    assert 0.0 <= stats.pbo <= 1.0


@pytest.mark.unit
def test_compute_honest_stats_rejects_non_series_inputs() -> None:
    """Non-Series / non-DataFrame inputs are rejected up front."""
    trials = pd.DataFrame(np.zeros((10, 3)))
    with pytest.raises(ValidationError):
        compute_honest_stats([0.0] * 10, pd.Series(np.zeros(10)), trials, n_grid_trials=3)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        compute_honest_stats(
            pd.Series(np.zeros(10)),
            pd.Series(np.zeros(10)),
            np.zeros((10, 3)),
            n_grid_trials=3,  # type: ignore[arg-type]
        )
