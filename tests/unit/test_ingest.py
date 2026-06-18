"""Tests for the offline ingestion group (extract + pushshift + reddit_api).

Covers:

- ticker-extraction correctness on text fixtures (cashtags, bare symbols, the
  ``"A"``/``"I"`` false-positive filter, length bounds, universe restriction),
- per-post de-duplication and ``created_utc`` retention,
- the batch ``extract_mention_table`` convenience wrapper,
- import purity (no praw / no network / no HTTP library at import time),
- the Pushshift adapter against a stubbed JSON getter (pagination, de-dupe,
  validation) with no real network,
- the Reddit/PRAW adapter against a stubbed lazy ``praw`` module (window filter,
  de-dupe, ``read_only``) with no real network,
- ``RedditCredentials.from_env`` happy path, redaction, and missing-var errors.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from typing import Any

import pytest

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.ingest import (
    FINANCE_STOPLIST,
    MentionExtraction,
    PushshiftQuery,
    RawPost,
    RedditCredentials,
    extract_mention_table,
    extract_mentions,
    fetch_pushshift_posts,
    fetch_reddit_posts,
)
from wsb_sentiment.ingest import pushshift as pushshift_mod


def _post(
    title: str = "",
    body: str = "",
    *,
    post_id: str = "p1",
    created_utc: int = 1_700_000_000,
) -> RawPost:
    """Build a minimal :class:`RawPost` for extraction tests."""
    return RawPost(
        post_id=post_id,
        created_utc=created_utc,
        title=title,
        body=body,
        author="someuser",
        score=1,
    )


# --------------------------------------------------------------------------- #
# extract: correctness                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cashtag_extraction() -> None:
    """``$TICKER`` cashtags are extracted, upper-cased, and flagged as cashtags."""
    result = extract_mentions(_post(title="loading up on $gme and $AMC today"))
    assert result.tickers == ("GME", "AMC")
    assert result.cashtag_tickers == ("GME", "AMC")


@pytest.mark.unit
def test_bare_symbol_extraction() -> None:
    """Bare uppercase symbols are extracted but not flagged as cashtags."""
    result = extract_mentions(_post(title="GME and TSLA to the moon"))
    assert set(result.tickers) == {"GME", "TSLA"}
    assert result.cashtag_tickers == ()


@pytest.mark.unit
def test_single_letter_false_positives_filtered() -> None:
    """Common single-letter tokens like ``A`` and ``I`` are NOT treated as tickers."""
    result = extract_mentions(_post(title="I think A B C are fine", body="I bought GME"))
    assert "A" not in result.tickers
    assert "I" not in result.tickers
    assert "GME" in result.tickers


@pytest.mark.unit
def test_finance_slang_stoplisted() -> None:
    """WSB slang / acronyms (YOLO, CEO, ETF, USA) are filtered as false positives."""
    text = "YOLO the CEO said the ETF in the USA is great, buy NVDA"
    result = extract_mentions(_post(title=text))
    assert "NVDA" in result.tickers
    for slang in ("YOLO", "CEO", "ETF", "USA"):
        assert slang in FINANCE_STOPLIST
        assert slang not in result.tickers


@pytest.mark.unit
def test_cashtag_overrides_stoplist() -> None:
    """An explicit ``$``-cashtag is kept even when the symbol is in the stoplist."""
    # "DD" is stoplisted as bare slang, but "$DD" (DuPont) is an explicit cashtag.
    result = extract_mentions(_post(title="check the DD then buy $DD"))
    assert "DD" in result.tickers
    assert "DD" in result.cashtag_tickers


@pytest.mark.unit
def test_length_bounds_enforced() -> None:
    """Symbols outside ``[min_symbol_len, max_symbol_len]`` are rejected."""
    # GOOGLE (6) exceeds the default max of 5; XY (2) is admissible.
    result = extract_mentions(_post(title="GOOGLE XY ZZZZZZ"))
    assert "GOOGLE" not in result.tickers
    assert "XY" in result.tickers


@pytest.mark.unit
def test_min_symbol_len_default_drops_single_letters() -> None:
    """The default ``min_symbol_len=2`` drops bare single letters even if not stoplisted."""
    # "Q" is a real ticker but a single letter; default bounds drop it.
    result = extract_mentions(_post(title="Q F T"))
    assert result.tickers == ()


@pytest.mark.unit
def test_universe_restriction_drops_outsiders() -> None:
    """With a universe, bare symbols outside it are dropped (PIT discipline)."""
    result = extract_mentions(
        _post(title="GME TSLA NVDA"),
        universe=["GME", "NVDA"],
    )
    assert set(result.tickers) == {"GME", "NVDA"}
    assert "TSLA" not in result.tickers


@pytest.mark.unit
def test_universe_restriction_applies_to_cashtags() -> None:
    """A supplied universe also bounds cashtags (no out-of-universe tradables)."""
    result = extract_mentions(
        _post(title="$GME $TSLA"),
        universe=["GME"],
    )
    assert result.tickers == ("GME",)
    assert result.cashtag_tickers == ("GME",)


@pytest.mark.unit
def test_dedupe_within_post() -> None:
    """A symbol repeated within a post appears exactly once."""
    result = extract_mentions(_post(title="GME GME GME", body="GME again $GME"))
    assert result.tickers.count("GME") == 1


@pytest.mark.unit
def test_created_utc_retained() -> None:
    """The originating ``created_utc`` is retained for the as-of cutoff."""
    result = extract_mentions(_post(title="GME", created_utc=1_690_000_123))
    assert result.created_utc == 1_690_000_123
    assert result.post_id == "p1"


@pytest.mark.unit
def test_title_scanned_before_body_ordering() -> None:
    """Mentions are ordered by first appearance, titles before bodies."""
    result = extract_mentions(_post(title="$AAPL", body="$MSFT"))
    assert result.tickers == ("AAPL", "MSFT")


@pytest.mark.unit
def test_numeric_dollar_not_a_cashtag() -> None:
    """``$1000`` is a price, not a cashtag, and is never extracted."""
    result = extract_mentions(_post(title="paid $1000 for GME"))
    assert result.cashtag_tickers == ()
    assert result.tickers == ("GME",)


@pytest.mark.unit
def test_lowercase_words_not_extracted() -> None:
    """Mixed-case and lowercase words are not bare-symbol candidates."""
    result = extract_mentions(_post(title="Tesla stock is going up apple too"))
    assert result.tickers == ()


@pytest.mark.unit
def test_custom_stoplist_overrides_default() -> None:
    """A caller-supplied stoplist replaces the default (case-insensitive)."""
    # With an empty stoplist, even YOLO survives as a bare symbol.
    result = extract_mentions(_post(title="YOLO GME"), stoplist=[])
    assert "YOLO" in result.tickers
    # A custom stoplist with "gme" (lowercase) drops GME.
    result2 = extract_mentions(_post(title="YOLO GME"), stoplist=["gme"])
    assert "GME" not in result2.tickers


@pytest.mark.unit
def test_duplicate_cashtag_kept_once() -> None:
    """A cashtag repeated within a post appears once in ``cashtag_tickers``."""
    result = extract_mentions(_post(title="$GME $GME", body="still $GME"))
    assert result.cashtag_tickers == ("GME",)


@pytest.mark.unit
def test_cashtag_out_of_length_bounds_dropped() -> None:
    """A cashtag whose length exceeds ``max_symbol_len`` is dropped."""
    # "$SIXSYM" is 6 letters: it matches the cashtag regex but fails the default
    # max_symbol_len=5 bound, exercising the length-guard ``continue``.
    result = extract_mentions(_post(title="$SIXSYM $GME"))
    assert result.cashtag_tickers == ("GME",)


@pytest.mark.unit
def test_empty_text_yields_no_mentions() -> None:
    """An empty post yields an empty, well-formed extraction."""
    result = extract_mentions(_post())
    assert result.tickers == ()
    assert result.cashtag_tickers == ()


@pytest.mark.unit
def test_mention_extraction_to_dict_roundtrip() -> None:
    """``MentionExtraction.to_dict`` returns JSON-friendly lists."""
    result = extract_mentions(_post(title="$GME TSLA"))
    payload = result.to_dict()
    assert isinstance(payload["tickers"], list)
    assert isinstance(payload["cashtag_tickers"], list)
    assert payload["created_utc"] == 1_700_000_000


# --------------------------------------------------------------------------- #
# extract: batch wrapper                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_extract_mention_table_drops_empty_posts() -> None:
    """The batch wrapper keeps only posts with at least one ticker, in order."""
    posts = [
        _post(title="just chatting, no tickers", post_id="a"),
        _post(title="$GME", post_id="b"),
        _post(title="YOLO CEO", post_id="c"),  # all stoplisted -> dropped
        _post(title="buy NVDA", post_id="d"),
    ]
    table = extract_mention_table(posts)
    assert [m.post_id for m in table] == ["b", "d"]
    assert all(isinstance(m, MentionExtraction) for m in table)


# --------------------------------------------------------------------------- #
# import purity                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_ingest_import_is_side_effect_free() -> None:
    """Importing the ingest subpackage pulls in no praw / no HTTP library.

    Runs in a FRESH subprocess so the assertion is not contaminated by modules
    already imported by the rest of the test session.
    """
    code = (
        "import sys; import wsb_sentiment.ingest; "
        "leaked = {'praw', 'httpx', 'requests'} & set(sys.modules); "
        "assert not leaked, sorted(leaked)"
    )
    # Strip coverage-subprocess hooks so the child interpreter does not re-start
    # coverage (which would double-load numpy's C extension under --cov).
    env = {k: v for k, v in os.environ.items() if not k.startswith("COVERAGE_")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, f"ingest import leaked network/praw libs: {proc.stderr}"


# --------------------------------------------------------------------------- #
# pushshift adapter (stubbed getter - no network)                            #
# --------------------------------------------------------------------------- #
def _record(rid: str, created: int, *, title: str = "GME", selftext: str = "") -> dict[str, Any]:
    return {
        "id": rid,
        "created_utc": created,
        "title": title,
        "selftext": selftext,
        "author": "u",
        "score": 3,
        "subreddit": "wallstreetbets",
    }


@pytest.mark.unit
def test_pushshift_pagination_and_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pager walks ``created_utc``, de-dupes, and stops on an empty page."""
    pages = [
        {"data": [_record("a", 1_600_000_100), _record("b", 1_600_000_200)]},
        # "b" repeats across the page boundary -> must be de-duplicated.
        {"data": [_record("b", 1_600_000_200), _record("c", 1_600_000_300)]},
        {"data": []},
    ]
    calls: list[dict[str, Any]] = []

    def _fake_getter(timeout: float) -> Any:
        def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
            calls.append(dict(params))
            return pages[len(calls) - 1]

        return _get

    monkeypatch.setattr(pushshift_mod, "_make_json_getter", _fake_getter)
    query = PushshiftQuery(
        subreddit="wallstreetbets",
        start=date(2020, 9, 1),
        end=date(2020, 9, 30),
        size=2,
    )
    posts = fetch_pushshift_posts(query)
    assert [p.post_id for p in posts] == ["a", "b", "c"]
    # created_utc retained and used to advance the cursor across pages.
    assert calls[1]["after"] == 1_600_000_200


@pytest.mark.unit
def test_pushshift_comment_kind_reads_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """For ``kind='comment'`` the body comes from the ``body`` field."""

    def _fake_getter(timeout: float) -> Any:
        def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
            assert "comment" in url
            return {"data": [{"id": "c1", "created_utc": 1_600_000_000, "body": "buy $GME"}]}

        return _get

    monkeypatch.setattr(pushshift_mod, "_make_json_getter", _fake_getter)
    query = PushshiftQuery(
        subreddit="wallstreetbets",
        start=date(2020, 9, 1),
        end=date(2020, 9, 2),
        kind="comment",
    )
    posts = fetch_pushshift_posts(query)
    assert posts[0].body == "buy $GME"
    assert posts[0].title == ""


@pytest.mark.unit
def test_pushshift_validation_errors() -> None:
    """Bad date range / kind / size / max_pages raise ValidationError."""
    base = dict(subreddit="wallstreetbets", start=date(2020, 9, 2), end=date(2020, 9, 1))
    with pytest.raises(ValidationError):
        fetch_pushshift_posts(PushshiftQuery(**base))  # end before start
    good = dict(subreddit="wallstreetbets", start=date(2020, 9, 1), end=date(2020, 9, 2))
    with pytest.raises(ValidationError):
        fetch_pushshift_posts(PushshiftQuery(kind="bogus", **good))
    with pytest.raises(ValidationError):
        fetch_pushshift_posts(PushshiftQuery(size=0, **good))
    with pytest.raises(ValidationError):
        fetch_pushshift_posts(PushshiftQuery(**good), max_pages=0)


@pytest.mark.unit
def test_pushshift_stops_when_cursor_does_not_advance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-second batch that fills the page does not loop forever."""
    page = {"data": [_record("a", 1_600_000_100), _record("b", 1_600_000_100)]}

    def _fake_getter(timeout: float) -> Any:
        def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
            return page  # always the same created_utc

        return _get

    monkeypatch.setattr(pushshift_mod, "_make_json_getter", _fake_getter)
    query = PushshiftQuery(
        subreddit="wallstreetbets",
        start=date(2020, 9, 1),
        end=date(2020, 9, 30),
        size=2,
    )
    posts = fetch_pushshift_posts(query, max_pages=1000)
    # First record's created_utc equals the cursor start -> loop terminates fast.
    assert [p.post_id for p in posts] == ["a", "b"]


@pytest.mark.unit
def test_pushshift_stops_on_partial_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page shorter than ``size`` ends pagination (no extra request)."""
    pages = [{"data": [_record("a", 1_600_000_100)]}]  # 1 record, size=2 -> partial
    calls: list[int] = []

    def _fake_getter(timeout: float) -> Any:
        def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
            calls.append(1)
            return pages[len(calls) - 1]

        return _get

    monkeypatch.setattr(pushshift_mod, "_make_json_getter", _fake_getter)
    query = PushshiftQuery(
        subreddit="wallstreetbets",
        start=date(2020, 9, 1),
        end=date(2020, 9, 30),
        size=2,
    )
    posts = fetch_pushshift_posts(query)
    assert [p.post_id for p in posts] == ["a"]
    assert len(calls) == 1  # stopped after the single partial page


@pytest.mark.unit
def test_pushshift_respects_max_pages_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every page is full and advances, the ``max_pages`` cap bounds the loop."""
    counter = {"n": 0}

    def _fake_getter(timeout: float) -> Any:
        def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
            counter["n"] += 1
            base = 1_600_000_000 + counter["n"] * 10
            # Always a full page (size=1) whose created_utc strictly advances,
            # so the loop only stops at the max_pages cap.
            return {"data": [_record(f"r{counter['n']}", base)]}

        return _get

    monkeypatch.setattr(pushshift_mod, "_make_json_getter", _fake_getter)
    query = PushshiftQuery(
        subreddit="wallstreetbets",
        start=date(2020, 9, 1),
        end=date(2020, 9, 30),
        size=1,
    )
    posts = fetch_pushshift_posts(query, max_pages=3)
    assert counter["n"] == 3
    assert [p.post_id for p in posts] == ["r1", "r2", "r3"]


@pytest.mark.unit
def test_raw_post_to_dict() -> None:
    """``RawPost.to_dict`` returns a plain serializable dict retaining created_utc."""
    payload = _post(title="GME", created_utc=1_650_000_000).to_dict()
    assert payload["created_utc"] == 1_650_000_000
    assert payload["subreddit"] == "wallstreetbets"
    assert isinstance(payload["extra"], dict)


@pytest.mark.unit
def test_pushshift_query_to_dict() -> None:
    """``PushshiftQuery.to_dict`` serializes dates to ISO strings."""
    payload = PushshiftQuery(
        subreddit="wallstreetbets", start=date(2021, 1, 1), end=date(2021, 1, 2)
    ).to_dict()
    assert payload["start"] == "2021-01-01"
    assert payload["end"] == "2021-01-02"


# --------------------------------------------------------------------------- #
# reddit_api adapter (stubbed lazy praw - no network)                        #
# --------------------------------------------------------------------------- #
class _FakeSubmission:
    def __init__(self, sid: str, created: float, title: str = "GME") -> None:
        self.id = sid
        self.created_utc = created
        self.title = title
        self.selftext = ""
        self.author = "u"
        self.score = 5
        self.subreddit = "wallstreetbets"


class _FakeSubredditEndpoint:
    def __init__(self, submissions: list[_FakeSubmission]) -> None:
        self._submissions = submissions

    def new(self, *, limit: int) -> list[_FakeSubmission]:
        return self._submissions[:limit]


class _FakeReddit:
    last_instance: _FakeReddit | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.read_only = False
        _FakeReddit.last_instance = self

    def subreddit(self, name: str) -> _FakeSubredditEndpoint:
        # Two in-window, one out-of-window (older than start).
        subs = [
            _FakeSubmission("s1", 1_600_000_100.0),
            _FakeSubmission("s2", 1_600_000_200.0),
            _FakeSubmission("s_old", 1_500_000_000.0),
        ]
        return _FakeSubredditEndpoint(subs)


def _install_fake_praw(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake = types.ModuleType("praw")
    fake.Reddit = _FakeReddit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "praw", fake)


@pytest.mark.unit
def test_fetch_reddit_posts_filters_window_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Posts outside ``[start, end]`` are dropped; client is read-only."""
    _install_fake_praw(monkeypatch)
    creds = RedditCredentials("cid", "secret", "ua/1.0")
    posts = fetch_reddit_posts(
        creds,
        start=date(2020, 9, 1),
        end=date(2020, 9, 30),
        limit=10,
    )
    assert [p.post_id for p in posts] == ["s1", "s2"]
    assert _FakeReddit.last_instance is not None
    assert _FakeReddit.last_instance.read_only is True


class _DupReddit(_FakeReddit):
    def subreddit(self, name: str) -> _FakeSubredditEndpoint:
        subs = [
            _FakeSubmission("s1", 1_600_000_100.0),
            _FakeSubmission("s1", 1_600_000_100.0),  # duplicate id -> de-duped
        ]
        return _FakeSubredditEndpoint(subs)


@pytest.mark.unit
def test_fetch_reddit_posts_dedupes_repeated_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submission id seen twice is emitted once."""
    import types

    fake = types.ModuleType("praw")
    fake.Reddit = _DupReddit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "praw", fake)
    creds = RedditCredentials("cid", "secret", "ua/1.0")
    posts = fetch_reddit_posts(creds, start=date(2020, 9, 1), end=date(2020, 9, 30))
    assert [p.post_id for p in posts] == ["s1"]


@pytest.mark.unit
def test_make_json_getter_builds_callable() -> None:
    """The lazy getter factory returns a callable without touching the network."""
    getter = pushshift_mod._make_json_getter(timeout=5.0)
    assert callable(getter)


@pytest.mark.unit
def test_make_json_getter_uses_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    """The built getter issues a stdlib ``urllib`` GET and parses JSON (no real net)."""
    import urllib.request

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data": [{"id": "x", "created_utc": 1}]}'

    captured: dict[str, Any] = {}

    def _fake_urlopen(url: str, timeout: float) -> _FakeResp:
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    getter = pushshift_mod._make_json_getter(timeout=7.0)
    payload = getter("https://api.example/search", {"subreddit": "wsb", "size": 2})
    assert payload == {"data": [{"id": "x", "created_utc": 1}]}
    assert "subreddit=wsb" in captured["url"]
    assert captured["timeout"] == 7.0


@pytest.mark.unit
def test_fetch_reddit_posts_validation() -> None:
    """Bad date range / limit raise ValidationError before importing praw."""
    creds = RedditCredentials("cid", "secret", "ua/1.0")
    with pytest.raises(ValidationError):
        fetch_reddit_posts(creds, start=date(2020, 9, 2), end=date(2020, 9, 1))
    with pytest.raises(ValidationError):
        fetch_reddit_posts(creds, start=date(2020, 9, 1), end=date(2020, 9, 2), limit=0)


# --------------------------------------------------------------------------- #
# RedditCredentials.from_env                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_credentials_from_env_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """All three env vars present -> credentials populated."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "ua/1.0")
    creds = RedditCredentials.from_env()
    assert creds.client_id == "cid"
    assert creds.client_secret == "csecret"
    assert creds.user_agent == "ua/1.0"


@pytest.mark.unit
def test_credentials_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/empty env var raises ValidationError naming the offender(s)."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("REDDIT_USER_AGENT", "ua/1.0")
    with pytest.raises(ValidationError, match="REDDIT_CLIENT_SECRET"):
        RedditCredentials.from_env()


@pytest.mark.unit
def test_credentials_to_dict_redacts_secret() -> None:
    """The secret is redacted in ``to_dict`` for safe logging."""
    payload = RedditCredentials("cid", "supersecret", "ua/1.0").to_dict()
    assert payload["client_secret"] == "***redacted***"
    assert payload["client_id"] == "cid"
    assert "supersecret" not in payload.values()
