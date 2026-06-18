"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`wsb_sentiment._rng.make_rng`) and
returns pandas objects, so tests across the suite share identical synthetic data
with known structure:

- ``synthetic_sentiment_panel`` - a per-(ticker, day) sentiment panel (mean
  compound, mention count, positive-share) plus a correlated-ish price panel.
- ``decaying_signal`` - a sentiment/return pair whose in-sample correlation decays
  out-of-sample (the honest null: an edge that fails DSR after costs).
- ``pure_noise`` - independent sentiment and returns with no relationship at all.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._rng import make_rng

_SEED = 20260617


def _ticker_labels(n: int) -> list[str]:
    """Return ``n`` deterministic ticker labels ``TKR00, TKR01, ...``."""
    return [f"TKR{i:02d}" for i in range(n)]


@dataclass(frozen=True, slots=True)
class SentimentPanel:
    """A seeded per-(ticker, day) sentiment panel paired with a price panel.

    Attributes
    ----------
    mean_compound:
        Wide ``day x ticker`` mean compound sentiment in ``[-1, 1]``.
    mention_count:
        Wide ``day x ticker`` daily mention count (non-negative integers).
    positive_share:
        Wide ``day x ticker`` positive-mention share in ``[0, 1]``.
    prices:
        Wide ``day x ticker`` strictly-positive synthetic prices.
    """

    mean_compound: pd.DataFrame
    mention_count: pd.DataFrame
    positive_share: pd.DataFrame
    prices: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DecayingSignal:
    """A sentiment/return pair with an in-sample edge that decays out-of-sample.

    Attributes
    ----------
    sentiment:
        Daily per-ticker sentiment (wide ``day x ticker``).
    returns:
        Daily per-ticker forward returns (wide ``day x ticker``).
    train_end:
        The in-sample/out-of-sample boundary date.
    """

    sentiment: pd.DataFrame
    returns: pd.DataFrame
    train_end: pd.Timestamp


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def synthetic_sentiment_panel() -> SentimentPanel:
    """A seeded per-(ticker, day) sentiment + price panel.

    Shape ``(504, 5)`` (about two trading years, five tickers). Sentiment is a
    bounded mean-reverting series, mention counts are Poisson, positive-share is a
    logistic transform of sentiment, and prices follow a mild common-factor GBM -
    all reproducible from a single seed.
    """
    gen = make_rng(_SEED)
    n_obs, n_assets = 504, 5
    index = pd.date_range("2021-01-04", periods=n_obs, freq="B")
    tickers = _ticker_labels(n_assets)

    # Bounded, mildly autocorrelated daily sentiment in [-1, 1].
    raw = gen.standard_normal((n_obs, n_assets))
    smoothed = pd.DataFrame(raw, index=index, columns=tickers).ewm(span=5).mean()
    mean_compound = np.tanh(smoothed.to_numpy() * 0.8)

    # Poisson mention counts with a per-ticker base rate.
    base_rate = gen.uniform(5.0, 40.0, size=n_assets)
    mention_count = gen.poisson(lam=base_rate, size=(n_obs, n_assets)).astype("float64")

    # Positive-share as a logistic function of the compound score.
    positive_share = 1.0 / (1.0 + np.exp(-3.0 * mean_compound))

    # Mild common-factor GBM prices.
    dt = 1.0 / 252.0
    sigma = gen.uniform(0.2, 0.5, size=n_assets)
    factor = gen.standard_normal(n_obs) * (0.12 * np.sqrt(dt))
    idio = gen.standard_normal((n_obs, n_assets))
    log_ret = sigma * np.sqrt(dt) * idio + np.outer(factor, gen.uniform(0.5, 1.3, n_assets))
    log_ret[0, :] = 0.0
    prices = gen.uniform(15.0, 250.0, n_assets) * np.exp(np.cumsum(log_ret, axis=0))

    return SentimentPanel(
        mean_compound=pd.DataFrame(mean_compound, index=index, columns=tickers),
        mention_count=pd.DataFrame(mention_count, index=index, columns=tickers),
        positive_share=pd.DataFrame(positive_share, index=index, columns=tickers),
        prices=pd.DataFrame(prices, index=index, columns=tickers),
    )


@pytest.fixture
def decaying_signal() -> DecayingSignal:
    """A sentiment/return pair with a DECAYING in-sample edge (the honest null).

    Returns are coupled to lagged sentiment with a coefficient that is positive
    early in the sample and decays to zero across the second half, so a leakage-free
    backtest finds a mild in-sample relationship that does not persist out-of-sample
    - the structure that fails the Deflated Sharpe after costs.
    """
    gen = make_rng(_SEED + 1)
    n_obs, n_assets = 504, 4
    index = pd.date_range("2021-01-04", periods=n_obs, freq="B")
    tickers = _ticker_labels(n_assets)

    sentiment = gen.standard_normal((n_obs, n_assets))
    # Coupling coefficient decays linearly from 0.10 (in-sample) to 0.0 (OOS).
    coupling = np.linspace(0.10, 0.0, n_obs).reshape(-1, 1)
    noise = gen.standard_normal((n_obs, n_assets)) * 0.02
    # Next-day return reacts to today's sentiment with the (decaying) coupling.
    lagged = np.vstack([np.zeros((1, n_assets)), sentiment[:-1]])
    returns = coupling * lagged * 0.01 + noise

    sent_df = pd.DataFrame(sentiment, index=index, columns=tickers)
    ret_df = pd.DataFrame(returns, index=index, columns=tickers)
    train_end = index[n_obs // 2]
    return DecayingSignal(sentiment=sent_df, returns=ret_df, train_end=train_end)


@pytest.fixture
def pure_noise() -> DecayingSignal:
    """Independent sentiment and returns: the no-relationship null.

    Sentiment and returns are drawn from independent generators with zero coupling,
    so any apparent in-sample association is pure sampling noise and the OOS edge is
    identically zero in expectation.
    """
    gen_s = make_rng(_SEED + 2)
    gen_r = make_rng(_SEED + 3)
    n_obs, n_assets = 504, 4
    index = pd.date_range("2021-01-04", periods=n_obs, freq="B")
    tickers = _ticker_labels(n_assets)

    sent_df = pd.DataFrame(gen_s.standard_normal((n_obs, n_assets)), index=index, columns=tickers)
    ret_df = pd.DataFrame(
        gen_r.standard_normal((n_obs, n_assets)) * 0.02, index=index, columns=tickers
    )
    train_end = index[n_obs // 2]
    return DecayingSignal(sentiment=sent_df, returns=ret_df, train_end=train_end)
