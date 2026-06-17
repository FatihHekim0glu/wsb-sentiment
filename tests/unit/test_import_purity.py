"""Import-purity and public-API smoke tests.

Asserts that importing ``wsb_sentiment`` (and the offline subpackages) pulls in no
heavy / network / model dependency, and that the curated public API is present.
"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.unit
def test_import_is_side_effect_free() -> None:
    """Importing the package must not import praw/torch/transformers/network libs."""
    import wsb_sentiment  # noqa: F401

    forbidden = {
        "praw",
        "torch",
        "transformers",
        "tensorflow",
        "vaderSentiment",
        "textblob",
        "plotly",
        "typer",
    }
    leaked = forbidden.intersection(sys.modules)
    assert not leaked, f"importing wsb_sentiment leaked heavy modules: {sorted(leaked)}"


@pytest.mark.unit
def test_public_api_is_exported() -> None:
    """A representative slice of the curated public API must be importable."""
    import wsb_sentiment as ws

    for name in (
        "__version__",
        "derive_verdict",
        "Verdict",
        "deflated_sharpe_ratio",
        "pbo_cscv",
        "newey_west_se",
        "generate_synthetic_panel",
        "SignalSpec",
        "rollup_daily_sentiment",
        "run_signal_backtest",
        "FINANCE_LEXICON",
    ):
        assert hasattr(ws, name), f"public API missing {name!r}"
    assert ws.__version__ == "0.1.0"


@pytest.mark.unit
def test_subpackages_import_cleanly() -> None:
    """Each subpackage must import without side effects."""
    import wsb_sentiment.aggregate
    import wsb_sentiment.backtest
    import wsb_sentiment.evaluation
    import wsb_sentiment.ingest
    import wsb_sentiment.nlp
    import wsb_sentiment.signal  # noqa: F401

    # Still no praw after importing the ingest subpackage explicitly.
    assert "praw" not in sys.modules
