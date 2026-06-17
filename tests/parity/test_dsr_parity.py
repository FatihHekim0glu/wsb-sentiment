"""Parity tests for the DSR/PSR honest-statistics primitives.

The Deflated and Probabilistic Sharpe ratios consumed by ``compute_honest_stats``
must agree with the reused :mod:`wsb_sentiment.evaluation.dsr` reference to
``1e-10``. We re-derive each statistic from its closed form independently and
assert equality, so any silent drift in the orchestration layer is caught.

We also check that:

- the DSR collapses to the plain PSR-vs-zero when ``n_trials == 1`` (the
  expected-maximum benchmark vanishes);
- the HAC t-stat helper equals ``mean / newey_west_se`` exactly;
- the PSR/DSR sit in ``[0, 1]`` and the DSR is monotone non-increasing in the
  multiplicity count.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._rng import make_rng
from wsb_sentiment.evaluation.dsr import (
    _EULER_MASCHERONI,
    _norm_cdf,
    _norm_ppf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from wsb_sentiment.evaluation.hac import newey_west_se
from wsb_sentiment.evaluation.stats import hac_tstat

_PARITY_TOL = 1e-10


def _psr_reference(sr: float, n_obs: int, skew: float, kurt: float, benchmark: float) -> float:
    """Independent closed-form PSR (Bailey-Lopez de Prado 2014, eq. 'PSR')."""
    variance = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr
    z = (sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return _norm_cdf(z)


def _dsr_benchmark_reference(n_trials: int, var_trials: float) -> float:
    """Independent expected-maximum benchmark Sharpe for the DSR."""
    if n_trials == 1 or var_trials == 0.0:
        return 0.0
    sqrt_v = math.sqrt(var_trials)
    n = float(n_trials)
    gamma = _EULER_MASCHERONI
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return sqrt_v * ((1.0 - gamma) * z1 + gamma * z2)


@pytest.mark.parity
@pytest.mark.parametrize("sr", [0.02, 0.05, 0.10, 0.18])
@pytest.mark.parametrize("n_obs", [126, 252, 504])
@pytest.mark.parametrize(("skew", "kurt"), [(0.0, 3.0), (-0.5, 5.0), (0.3, 4.2)])
def test_psr_matches_independent_reference(sr: float, n_obs: int, skew: float, kurt: float) -> None:
    """PSR vs zero matches the independent closed form to 1e-10."""
    got = probabilistic_sharpe_ratio(sr, n_obs=n_obs, skew=skew, kurtosis=kurt)
    want = _psr_reference(sr, n_obs, skew, kurt, benchmark=0.0)
    assert abs(got - want) < _PARITY_TOL
    assert 0.0 <= got <= 1.0


@pytest.mark.parity
@pytest.mark.parametrize("n_trials", [1, 5, 20, 100, 500])
@pytest.mark.parametrize("var_trials", [0.0, 0.005, 0.02])
def test_dsr_matches_psr_with_expected_max_benchmark(n_trials: int, var_trials: float) -> None:
    """DSR equals PSR evaluated against the expected-maximum benchmark (1e-10)."""
    sr, n_obs, skew, kurt = 0.12, 504, -0.2, 4.0
    got = deflated_sharpe_ratio(
        sr,
        n_obs=n_obs,
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
        skew=skew,
        kurtosis=kurt,
    )
    benchmark = _dsr_benchmark_reference(n_trials, var_trials)
    want = _psr_reference(sr, n_obs, skew, kurt, benchmark=benchmark)
    assert abs(got - want) < _PARITY_TOL
    assert 0.0 <= got <= 1.0


@pytest.mark.parity
def test_dsr_reduces_to_psr_when_single_trial() -> None:
    """With one trial the DSR is exactly the plain PSR-vs-zero (1e-10)."""
    sr, n_obs = 0.15, 252
    dsr = deflated_sharpe_ratio(sr, n_obs=n_obs, n_trials=1, variance_of_trial_sharpes=0.03)
    psr = probabilistic_sharpe_ratio(sr, n_obs=n_obs)
    assert abs(dsr - psr) < _PARITY_TOL


@pytest.mark.parity
def test_dsr_monotone_non_increasing_in_trials() -> None:
    """The DSR never rises as the multiplicity count grows."""
    sr, n_obs, var_trials = 0.13, 504, 0.01
    prev = 1.0
    for n_trials in (1, 2, 5, 20, 50, 200, 1000):
        dsr = deflated_sharpe_ratio(
            sr,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=var_trials,
        )
        assert dsr <= prev + 1e-12
        prev = dsr


@pytest.mark.parity
def test_hac_tstat_matches_mean_over_newey_west_se() -> None:
    """The HAC t-stat helper equals mean / newey_west_se to 1e-10."""
    gen = make_rng(11)
    # Mildly autocorrelated returns so the HAC correction actually bites.
    raw = gen.standard_normal(400)
    ar = np.empty_like(raw)
    ar[0] = raw[0]
    for i in range(1, raw.size):
        ar[i] = 0.3 * ar[i - 1] + raw[i]
    returns = pd.Series(ar * 0.01 + 0.0003)

    tstat, pvalue = hac_tstat(returns)
    se = newey_west_se(returns)
    expected_t = float(returns.to_numpy().mean()) / se
    assert abs(tstat - expected_t) < _PARITY_TOL

    expected_p = 2.0 * (1.0 - _norm_cdf(abs(expected_t)))
    assert abs(pvalue - expected_p) < _PARITY_TOL
    assert 0.0 <= pvalue <= 1.0
