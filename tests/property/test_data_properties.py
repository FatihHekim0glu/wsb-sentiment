"""Property-based tests for the synthetic generator's structural invariants.

These assert the honest-null structure and the no-lookahead-relevant invariants of
:mod:`wsb_sentiment.data` hold across a range of seeds and ticker counts:

- the in-sample LAGGED sentiment-to-return correlation decays out-of-sample on
  average and for the large majority of seeds (the honest null by construction;
  see :func:`test_edge_decays_on_average` for why this is a statistical, not a
  per-seed, claim);
- value domains (bounded compound, [0, 1] positive-share, non-negative integer
  mentions, strictly-positive prices) hold for every seed;
- the PIT universe mask is monotone non-decreasing per ticker (no future-driven
  membership removal);
- ``compute_returns`` never lengthens the panel and never forward-fills gaps.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wsb_sentiment.data import compute_returns, generate_synthetic_panel, pit_universe

_START = date(2021, 1, 4)
_END = date(2022, 12, 30)
_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _tickers(n: int) -> list[str]:
    return [f"SYN{i:02d}" for i in range(n)]


def _lag_corr(sentiment: pd.DataFrame, returns: pd.DataFrame, sl: slice) -> float:
    s = sentiment.iloc[sl].shift(1).to_numpy().ravel()
    r = returns.iloc[sl].to_numpy().ravel()
    mask = np.isfinite(s) & np.isfinite(r)
    return float(np.corrcoef(s[mask], r[mask])[0, 1])


def _is_oos_lag_corr(seed: int, n_assets: int) -> tuple[float, float]:
    """Return the (in-sample, out-of-sample) lagged sentiment-return correlation."""
    panel = generate_synthetic_panel(_tickers(n_assets), _START, _END, seed=seed)
    returns = panel.prices.pct_change(fill_method=None)
    n = len(panel.mean_compound)
    half = n // 2
    in_sample = _lag_corr(panel.mean_compound, returns, slice(0, half))
    out_of_sample = _lag_corr(panel.mean_compound, returns, slice(half, n))
    return in_sample, out_of_sample


def test_edge_decays_on_average() -> None:
    """The lagged edge decays out-of-sample on average and for most seeds.

    The generator ramps the lagged sentiment-to-return coupling from
    ``in_sample_corr`` down to ``oos_corr = 0`` by the train/test boundary, so the
    in-sample half carries predictive power that the out-of-sample half does not.
    On a finite, noisy synthetic panel a single seed can still show an out-of-sample
    correlation slightly above the in-sample one by chance (an earlier strict
    per-seed assertion failed on, for example, seed 82 by about 0.001). The honest
    claim is therefore statistical: averaged over a fixed sweep of seeds the
    in-sample correlation is the larger, it is positive, and the decay holds for the
    large majority of individual seeds.
    """
    seeds = range(120)
    n_assets = 5
    in_samples: list[float] = []
    out_samples: list[float] = []
    decayed = 0
    for seed in seeds:
        in_sample, out_of_sample = _is_oos_lag_corr(seed, n_assets)
        in_samples.append(in_sample)
        out_samples.append(out_of_sample)
        if abs(out_of_sample) < abs(in_sample):
            decayed += 1

    in_arr = np.asarray(in_samples)
    out_arr = np.asarray(out_samples)

    # Every seed shows a positive in-sample lagged correlation by construction.
    assert (in_arr > 0.0).all()
    # On average the in-sample correlation clearly exceeds the out-of-sample one.
    assert float(in_arr.mean()) > float(np.abs(out_arr).mean())
    # The decay holds for the large majority of individual seeds.
    assert decayed >= int(0.8 * len(in_samples))


@_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=5000), n_assets=st.integers(min_value=3, max_value=8)
)
def test_in_sample_lag_corr_is_positive_for_every_seed(seed: int, n_assets: int) -> None:
    """The in-sample lagged sentiment-return correlation is positive for any seed.

    This is the per-seed half of the honest null that does hold strictly: the early
    sample always carries a positive lagged edge. Its out-of-sample decay is a
    statistical claim covered by :func:`test_edge_decays_on_average`.
    """
    in_sample, _ = _is_oos_lag_corr(seed, n_assets)
    assert in_sample > 0.0


@_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=5000), n_assets=st.integers(min_value=2, max_value=8)
)
def test_value_domains_hold(seed: int, n_assets: int) -> None:
    """Compound, positive-share, mention-count, and price domains hold for any seed."""
    panel = generate_synthetic_panel(_tickers(n_assets), _START, date(2021, 12, 31), seed=seed)
    mc = panel.mean_compound.to_numpy()
    ps = panel.positive_share.to_numpy()
    mn = panel.mention_count.to_numpy()
    assert mc.min() >= -1.0 and mc.max() <= 1.0
    assert ps.min() >= 0.0 and ps.max() <= 1.0
    assert mn.min() >= 0.0 and np.allclose(mn, np.round(mn))
    assert panel.prices.to_numpy().min() > 0.0


@_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=5000), n_assets=st.integers(min_value=2, max_value=8)
)
def test_pit_mask_is_monotone(seed: int, n_assets: int) -> None:
    """Once admitted, a ticker stays in the universe (no future-driven removal)."""
    sessions = pd.date_range(_START, date(2021, 12, 31), freq="B")
    mask = pit_universe(_tickers(n_assets), sessions)
    for ticker in mask.columns:
        col = mask[ticker].to_numpy().astype(int)
        assert (np.diff(col) >= 0).all()


@_SETTINGS
@given(seed=st.integers(min_value=0, max_value=5000))
def test_compute_returns_never_lengthens_or_ffills(seed: int) -> None:
    """Returns drop exactly the leading row and never manufacture across gaps."""
    panel = generate_synthetic_panel(_tickers(4), _START, date(2021, 12, 31), seed=seed)
    returns = compute_returns(panel.prices)
    assert len(returns) == len(panel.prices) - 1
    # Identical to a no-ffill pct_change with the leading NaN row dropped.
    reference = panel.prices.pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_frame_equal(returns, reference)
