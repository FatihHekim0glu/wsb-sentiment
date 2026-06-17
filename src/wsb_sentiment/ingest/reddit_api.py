"""Reddit/PRAW offline batch ingestion adapter (LAZY client).

The live Reddit path (via PRAW) supplements the Pushshift archive for recent
r/wallstreetbets activity in the real-data ``ingest``+``score`` CLI path. Like
the Pushshift adapter this is an OFFLINE BATCH tool and is never called at
request time by the deployed API.

IMPORT PURITY: ``praw`` lives behind the ``[ingest]`` extra and is imported
LAZILY inside :func:`fetch_reddit_posts`. Importing this module triggers no praw
import and no network call.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from wsb_sentiment.ingest.pushshift import RawPost


@dataclass(frozen=True, slots=True)
class RedditCredentials:
    """OAuth credentials for the Reddit API (read-only script app).

    Attributes
    ----------
    client_id:
        The Reddit application client id.
    client_secret:
        The Reddit application client secret.
    user_agent:
        A descriptive user-agent string (required by the Reddit API).
    """

    client_id: str
    client_secret: str
    user_agent: str

    @classmethod
    def from_env(cls) -> RedditCredentials:
        """Build credentials from ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` / ``REDDIT_USER_AGENT``.

        Returns
        -------
        RedditCredentials
            Credentials populated from the environment.

        Raises
        ------
        NotImplementedError
            This is a typed stub awaiting implementation.

        Notes
        -----
        The implementation reads ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` /
        ``REDDIT_USER_AGENT`` from the environment LAZILY (``import os`` inside the
        body), so importing this module requires none of them.
        """
        raise NotImplementedError("RedditCredentials.from_env is not yet implemented")

    def to_dict(self) -> dict[str, str]:
        """Return a plain ``dict`` with the secret REDACTED for safe logging."""
        return {
            "client_id": self.client_id,
            "client_secret": "***redacted***",
            "user_agent": self.user_agent,
        }


def fetch_reddit_posts(
    credentials: RedditCredentials,
    *,
    subreddit: str = "wallstreetbets",
    start: date,
    end: date,
    limit: int = 1000,
) -> list[RawPost]:
    """Fetch recent posts via PRAW over ``[start, end]`` (OFFLINE BATCH).

    Constructs a read-only PRAW client LAZILY (``import praw`` happens inside this
    function) and returns the same :class:`RawPost` shape as the Pushshift
    adapter, retaining ``created_utc`` for the as-of cutoff.

    Parameters
    ----------
    credentials:
        The Reddit OAuth credentials.
    subreddit:
        The subreddit to pull from.
    start, end:
        Inclusive backfill date range.
    limit:
        A hard cap on the number of posts fetched.

    Returns
    -------
    list[RawPost]
        The fetched raw posts, de-duplicated by ``post_id``.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("fetch_reddit_posts is not yet implemented")
