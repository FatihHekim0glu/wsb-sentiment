"""Pushshift offline batch ingestion adapter (LAZY HTTP client).

Pushshift is the historical r/wallstreetbets archive used to backfill posts and
comments for the real-data ``ingest``+``score`` CLI path. This module is an
OFFLINE BATCH tool: it is never called at request time by the deployed API.

IMPORT PURITY: no network call, no HTTP-library import at module import time. The
HTTP client is constructed lazily inside :func:`fetch_pushshift_posts`. Pushshift
coverage has a well-documented deletion bias (removed/deleted posts are missing)
and a coverage gap after 2023; both are surfaced in the README limitations.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True, slots=True)
class PushshiftQuery:
    """A single Pushshift backfill request over a date range.

    Attributes
    ----------
    subreddit:
        The subreddit to pull from (default ``"wallstreetbets"``).
    start:
        Inclusive start date of the backfill window.
    end:
        Inclusive end date of the backfill window.
    kind:
        Either ``"submission"`` (posts) or ``"comment"``.
    size:
        Page size for each paginated request.
    """

    subreddit: str
    start: date
    end: date
    kind: str = "submission"
    size: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this query."""
        out = asdict(self)
        out["start"] = self.start.isoformat()
        out["end"] = self.end.isoformat()
        return out


@dataclass(frozen=True, slots=True)
class RawPost:
    """A single raw post/comment fetched from Pushshift (or Reddit).

    Attributes
    ----------
    post_id:
        The platform-unique id (``id`` or ``fullname``); used for de-duplication.
    created_utc:
        The epoch-seconds creation timestamp (RETAINED for the as-of cutoff).
    title:
        The submission title (empty for comments).
    body:
        The selftext / comment body.
    author:
        The author handle (``"[deleted]"`` when removed).
    score:
        The net upvote score at fetch time.
    subreddit:
        The originating subreddit.
    """

    post_id: str
    created_utc: int
    title: str
    body: str
    author: str
    score: int
    subreddit: str = "wallstreetbets"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this raw post."""
        return asdict(self)


def fetch_pushshift_posts(
    query: PushshiftQuery,
    *,
    base_url: str = "https://api.pushshift.io",
    timeout: float = 30.0,
    max_pages: int = 1000,
) -> list[RawPost]:
    """Fetch raw posts/comments from Pushshift over ``query``'s date range (OFFLINE BATCH).

    Paginates the Pushshift endpoint by ``created_utc``, retaining the raw
    ``created_utc`` on every record so the downstream aggregator can apply its
    strict as-of cutoff at the prior session close. The HTTP client is imported
    LAZILY inside this function so importing the module triggers no network and
    no third-party import.

    Parameters
    ----------
    query:
        The backfill request (subreddit, date range, kind, page size).
    base_url:
        The Pushshift API base URL.
    timeout:
        Per-request timeout in seconds.
    max_pages:
        A hard pagination cap guarding against runaway backfills.

    Returns
    -------
    list[RawPost]
        The fetched raw posts/comments, de-duplicated by ``post_id``.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("fetch_pushshift_posts is not yet implemented")
