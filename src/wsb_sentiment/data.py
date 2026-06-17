"""Synthetic sentiment + price generator and data loaders.

The shipped DEFAULT (tests and the deployed tool) runs on a SYNTHETIC generator —
no Reddit/Pushshift/Polygon keys required — constructed so the in-sample
sentiment-return correlation DECAYS out-of-sample and FAILS the Deflated Sharpe
after costs. That is the honest null BY CONSTRUCTION: there is a mild early
relationship dominated by contemporaneous attention/return feedback that does not
persist, so a leakage-free backtest cannot extract a durable edge.

The generator emits, per (ticker, day):

- a daily sentiment panel (mean compound score, mention count, positive-share),
- a correlated-ish price/return panel,

all seeded via :func:`wsb_sentiment._rng.make_rng` so a given ``(tickers, start,
end, seed)`` reproduces byte-identical data.

Real-data loaders (cached parquet sentiment, Polygon prices) are provided for the
ingest path; they import their heavy dependencies LAZILY. A point-in-time
universe hook restricts the tradable set to a PIT-consistent S&P-500 membership so
no future constituent is selected.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import pandas as pd

from wsb_sentiment._typing import PricesLike

#: Where a sentiment/price panel ultimately came from. Returned alongside the
#: data so callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "cache", "synthetic"]

# quantcore-candidate: synthetic structure mirrors hrp-portfolio:src/hrp/data.py
# (seeded GBM prices) extended with a decaying sentiment-return coupling.


@dataclass(frozen=True, slots=True)
class SyntheticPanel:
    """A synthetic sentiment + price bundle with honest-null structure.

    Attributes
    ----------
    mean_compound:
        Wide ``day x ticker`` panel of the daily mean compound sentiment.
    mention_count:
        Wide ``day x ticker`` panel of the daily mention count.
    positive_share:
        Wide ``day x ticker`` panel of the daily positive-mention share.
    prices:
        Wide ``day x ticker`` panel of synthetic close prices.
    universe_mask:
        Wide ``day x ticker`` boolean point-in-time tradability mask.
    data_source:
        Always ``"synthetic"`` for this generator.
    """

    mean_compound: pd.DataFrame
    mention_count: pd.DataFrame
    positive_share: pd.DataFrame
    prices: pd.DataFrame
    universe_mask: pd.DataFrame
    data_source: DataSource = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _panel(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
            is_bool: dict[str, bool] = {
                str(c): bool(pd.api.types.is_bool_dtype(df[c])) for c in df.columns
            }
            return {
                str(idx): {
                    str(c): (None if pd.isna(v) else (bool(v) if is_bool[str(c)] else float(v)))
                    for c, v in row.items()
                }
                for idx, row in df.iterrows()
            }

        return {
            "mean_compound": _panel(self.mean_compound),
            "mention_count": _panel(self.mention_count),
            "positive_share": _panel(self.positive_share),
            "prices": _panel(self.prices),
            "universe_mask": _panel(self.universe_mask),
            "data_source": str(self.data_source),
            "meta": dict(self.meta),
        }


def generate_synthetic_panel(
    tickers: list[str],
    start: date,
    end: date,
    *,
    seed: int = 7,
    in_sample_corr: float = 0.08,
    oos_corr: float = 0.0,
    mention_lambda: float = 25.0,
) -> SyntheticPanel:
    """Generate a seeded synthetic sentiment + price panel with a DECAYING edge.

    The generator couples next-day returns to today's sentiment with a coefficient
    that is ``in_sample_corr`` early in the sample and decays toward ``oos_corr``
    over time, while a stronger CONTEMPORANEOUS attention/return feedback term
    confounds the relationship. The result: a mild in-sample correlation that does
    not survive out-of-sample, failing the Deflated Sharpe after costs — the
    honest null by construction.

    Parameters
    ----------
    tickers:
        The ticker symbols to simulate.
    start, end:
        Inclusive date range (business days).
    seed:
        Master RNG seed; ``(tickers, start, end, seed)`` reproduces the panel.
    in_sample_corr:
        The early-sample sentiment -> next-day-return coupling coefficient.
    oos_corr:
        The late-sample coupling the relationship decays toward.
    mention_lambda:
        The base Poisson rate for daily per-ticker mention counts.

    Returns
    -------
    SyntheticPanel
        The seeded sentiment + price bundle.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("generate_synthetic_panel is not yet implemented")


def load_sentiment_panel(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source_pref: Literal["auto", "cache", "synthetic"] = "synthetic",
    seed: int = 7,
    cache_path: str | None = None,
) -> tuple[SyntheticPanel, DataSource]:
    """Load the daily sentiment + price panel for the request (synthetic default).

    Resolution order: a precomputed cached parquet (``"cache"``) when available,
    else the deterministic :func:`generate_synthetic_panel` (``"synthetic"``). The
    deployed default is ``"synthetic"`` so the result is reproducible and the null
    is honest; no live Pushshift/PRAW ingestion or VADER scoring happens here.

    LAZY IMPORT: the parquet/diskcache reader (the ``data`` extra) is imported
    inside this function, never at module import time.

    Parameters
    ----------
    tickers:
        The ticker symbols to load.
    start, end:
        Inclusive date range.
    source_pref:
        Preferred source; ``"synthetic"`` (the deployed default) always succeeds.
    seed:
        Master RNG seed forwarded to the synthetic generator.
    cache_path:
        Optional path to a precomputed sentiment parquet.

    Returns
    -------
    tuple[SyntheticPanel, DataSource]
        The loaded panel and the source it came from.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("load_sentiment_panel is not yet implemented")


def compute_returns(prices: PricesLike) -> pd.DataFrame:
    r"""Convert a price panel to forward-safe simple returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled
    before differencing, because ffill-then-diff manufactures spurious zero returns
    across gaps and leaks information. The leading all-NaN row is dropped.

    Parameters
    ----------
    prices:
        A wide panel of prices (rows = date, columns = ticker).

    Returns
    -------
    pandas.DataFrame
        Simple returns with the leading NaN row removed.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("compute_returns is not yet implemented")


def pit_universe(
    tickers: list[str],
    sessions: pd.DatetimeIndex,
    *,
    source_pref: Literal["auto", "synthetic"] = "synthetic",
) -> pd.DataFrame:
    """Return a point-in-time tradability mask (no future-constituent selection).

    Builds a wide ``day x ticker`` boolean mask of S&P-500 membership AS-OF each
    date so the tradable signal never selects a symbol on the basis of FUTURE index
    membership. The real path consults the Polygon S&P-500 universe; the synthetic
    default simulates a PIT-consistent membership set. Meme tickers outside the
    universe are descriptive-only (mask ``False``) and never traded.

    Parameters
    ----------
    tickers:
        The candidate ticker symbols.
    sessions:
        The trading-session calendar to build the mask over.
    source_pref:
        Preferred universe source; ``"synthetic"`` simulates a PIT membership.

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` boolean tradability mask.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("pit_universe is not yet implemented")
