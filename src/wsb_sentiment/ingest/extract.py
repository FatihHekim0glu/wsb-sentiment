"""Ticker-mention extraction from raw WSB text.

Extracts ticker references from post titles/bodies: ``$TICKER`` cashtags and bare
uppercase symbols (filtered against a common-word stoplist to suppress false
positives like ``"YOLO"`` or ``"CEO"``). Mentions are de-duplicated per post and
the originating ``created_utc`` is retained so the aggregator can apply its
strict as-of cutoff.

Importing this module has no side effects (the regexes are compiled at import,
which is pure; no network, no praw).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from wsb_sentiment.ingest.pushshift import RawPost


@dataclass(frozen=True, slots=True)
class MentionExtraction:
    """The tickers mentioned in a single raw post, with provenance retained.

    Attributes
    ----------
    post_id:
        The originating post id (for de-duplication and traceability).
    created_utc:
        The epoch-seconds creation timestamp (RETAINED for the as-of cutoff).
    tickers:
        The distinct, upper-cased tickers mentioned in this post.
    cashtag_tickers:
        The subset that appeared with an explicit ``$`` cashtag prefix (a
        higher-precision signal than a bare uppercase token).
    """

    post_id: str
    created_utc: int
    tickers: tuple[str, ...]
    cashtag_tickers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this extraction."""
        out = asdict(self)
        out["tickers"] = list(self.tickers)
        out["cashtag_tickers"] = list(self.cashtag_tickers)
        return out


def extract_mentions(
    post: RawPost,
    *,
    universe: Sequence[str] | None = None,
    stoplist: Iterable[str] | None = None,
    min_symbol_len: int = 2,
    max_symbol_len: int = 5,
) -> MentionExtraction:
    """Extract distinct ticker mentions from a single raw post.

    Scans ``post.title`` and ``post.body`` for ``$TICKER`` cashtags and bare
    uppercase symbols of length ``[min_symbol_len, max_symbol_len]``, drops tokens
    in ``stoplist`` (common all-caps slang/words), optionally restricts to
    ``universe``, de-duplicates, and retains ``created_utc``.

    Parameters
    ----------
    post:
        The raw post/comment to scan.
    universe:
        Optional whitelist of admissible tickers; when given, bare-symbol matches
        outside the universe are dropped (cashtags may be retained as descriptive).
    stoplist:
        All-caps words to suppress (e.g. ``"YOLO"``, ``"CEO"``, ``"USA"``).
    min_symbol_len, max_symbol_len:
        Inclusive length bounds on a bare-symbol candidate.

    Returns
    -------
    MentionExtraction
        The distinct tickers and cashtag subset for this post.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("extract_mentions is not yet implemented")
