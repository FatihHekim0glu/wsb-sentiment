"""Unit tests for the reused honest-stats primitives and the verdict truth table.

These exercise the *copied* (already-implemented) infra — DSR/PSR, the verdict
purity rule, and the seeded fixtures — so the suite has real green coverage even
while the new compute kernels remain stubs.
"""

from __future__ import annotations

import pytest

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from wsb_sentiment.evaluation.verdict import Verdict, derive_verdict


@pytest.mark.unit
def test_psr_in_unit_interval() -> None:
    """The PSR is a probability in [0, 1] and rises with the observed Sharpe."""
    low = probabilistic_sharpe_ratio(0.05, n_obs=252)
    high = probabilistic_sharpe_ratio(0.20, n_obs=252)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


@pytest.mark.unit
def test_dsr_non_increasing_in_n_trials() -> None:
    """The Deflated Sharpe is non-increasing in the trial count (multiplicity)."""
    few = deflated_sharpe_ratio(0.15, n_obs=504, n_trials=1, variance_of_trial_sharpes=0.01)
    many = deflated_sharpe_ratio(0.15, n_obs=504, n_trials=200, variance_of_trial_sharpes=0.01)
    assert 0.0 <= many <= few <= 1.0


@pytest.mark.unit
def test_verdict_false_when_any_hurdle_fails() -> None:
    """signal_has_edge must be False unless ALL hurdles pass simultaneously."""
    # All hurdles pass -> True.
    good = derive_verdict(0.8, deflated_sharpe=0.99, pbo=0.1, hac_pvalue=0.01)
    assert isinstance(good, Verdict)
    assert good.signal_has_edge is True

    # Each single failure flips the verdict to False.
    assert derive_verdict(0.0, 0.99, 0.1, 0.01).signal_has_edge is False  # Sharpe
    assert derive_verdict(0.8, 0.50, 0.1, 0.01).signal_has_edge is False  # DSR
    assert derive_verdict(0.8, 0.99, 0.9, 0.01).signal_has_edge is False  # PBO
    assert derive_verdict(0.8, 0.99, 0.1, 0.50).signal_has_edge is False  # HAC


@pytest.mark.unit
def test_verdict_rejects_out_of_range_inputs() -> None:
    """Out-of-[0,1] probability inputs raise ValidationError."""
    with pytest.raises(ValidationError):
        derive_verdict(0.8, deflated_sharpe=1.5, pbo=0.1, hac_pvalue=0.01)
    with pytest.raises(ValidationError):
        derive_verdict(0.8, deflated_sharpe=0.99, pbo=0.1, hac_pvalue=2.0)


@pytest.mark.unit
def test_verdict_to_dict_round_trips() -> None:
    """The verdict serializes to a JSON-friendly dict with list reasons."""
    v = derive_verdict(0.1, 0.4, 0.6, 0.3)
    payload = v.to_dict()
    assert payload["signal_has_edge"] is False
    assert isinstance(payload["reasons"], list)
    assert len(payload["reasons"]) == 4


@pytest.mark.unit
def test_fixtures_have_expected_shape(
    synthetic_sentiment_panel: object,
    decaying_signal: object,
    pure_noise: object,
) -> None:
    """The seeded fixtures load deterministically with aligned shapes."""
    from tests.conftest import DecayingSignal, SentimentPanel

    assert isinstance(synthetic_sentiment_panel, SentimentPanel)
    assert synthetic_sentiment_panel.mean_compound.shape == (504, 5)
    assert synthetic_sentiment_panel.prices.shape == (504, 5)

    assert isinstance(decaying_signal, DecayingSignal)
    assert decaying_signal.sentiment.shape == decaying_signal.returns.shape

    assert isinstance(pure_noise, DecayingSignal)
    assert pure_noise.sentiment.shape == (504, 4)
