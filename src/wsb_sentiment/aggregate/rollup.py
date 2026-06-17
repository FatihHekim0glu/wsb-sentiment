"""Per-(ticker, day) sentiment roll-up under a strict as-of cutoff.

Given per-post sentiment scores joined to ticker mentions (each carrying a raw
``created_utc``), this module aggregates to a daily per-ticker panel:

- mean compound, median compound,
- mention count,
- positive-share (fraction of mentions with compound > 0).

STRICT AS-OF CUTOFF (leakage guard): every post is assigned to the trading day
whose signal it is allowed to inform, where the cutoff is the PRIOR session
close. A post created after a session's close rolls into the NEXT session's
sentiment, never the current one. This is the aggregation-side half of the
no-lookahead discipline; the signal layer adds ``signal.shift(1)`` on top.

The aggregator is required to be PREFIX-DETERMINISTIC: appending future posts must
not change any already-emitted (ticker, day) row (property-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class DailyRollup:
    """A daily per-ticker sentiment panel plus its aggregation provenance.

    Attributes
    ----------
    mean_compound:
        Wide ``day x ticker`` panel of the mean compound score.
    median_compound:
        Wide ``day x ticker`` panel of the median compound score.
    mention_count:
        Wide ``day x ticker`` panel of the mention count (integer-valued floats).
    positive_share:
        Wide ``day x ticker`` panel of the positive-mention share in ``[0, 1]``.
    session_tz:
        The exchange timezone used to define the session-close cutoff.
    """

    mean_compound: pd.DataFrame
    median_compound: pd.DataFrame
    mention_count: pd.DataFrame
    positive_share: pd.DataFrame
    session_tz: str = "America/New_York"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _panel(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
            return {
                str(idx): {
                    str(c): (None if pd.isna(v) else float(v)) for c, v in row.items()
                }
                for idx, row in df.iterrows()
            }

        return {
            "mean_compound": _panel(self.mean_compound),
            "median_compound": _panel(self.median_compound),
            "mention_count": _panel(self.mention_count),
            "positive_share": _panel(self.positive_share),
            "session_tz": str(self.session_tz),
            "meta": dict(self.meta),
        }


def rollup_daily_sentiment(
    scored_mentions: pd.DataFrame,
    *,
    session_close: str = "16:00",
    session_tz: str = "America/New_York",
    sessions: pd.DatetimeIndex | None = None,
) -> DailyRollup:
    """Aggregate scored ticker mentions to a daily per-ticker panel (as-of cutoff).

    Parameters
    ----------
    scored_mentions:
        A long table with at least the columns ``created_utc`` (epoch seconds),
        ``ticker`` (upper-cased symbol), and ``compound`` (VADER compound score).
    session_close:
        The exchange session-close wall-clock time (``"HH:MM"``); posts after this
        time roll into the NEXT trading session (the strict as-of cutoff).
    session_tz:
        The exchange timezone in which ``session_close`` is interpreted.
    sessions:
        Optional explicit trading-session calendar; when ``None`` a business-day
        calendar spanning the observed dates is used.

    Returns
    -------
    DailyRollup
        The wide daily per-ticker panels (mean/median compound, mention count,
        positive-share) under the as-of cutoff.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("rollup_daily_sentiment is not yet implemented")
