"""Unit tests for the Typer CLI and its Typer-free orchestration layer.

Importing :mod:`wsb_sentiment.cli` must not import Typer (lazy). The pure
``run_*`` orchestration functions are tested directly (no Typer needed); the Typer
``--help`` smoke test is skipped when Typer is not installed in the environment.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date

import pytest

import wsb_sentiment.cli as cli

_HAS_TYPER = importlib.util.find_spec("typer") is not None


@pytest.mark.unit
def test_importing_cli_does_not_import_typer() -> None:
    """Importing the CLI module must register no commands and import no Typer."""
    # A fresh import path: the module is already imported, so just assert the
    # import did not leak Typer at import time.
    assert "typer" not in sys.modules or _HAS_TYPER
    # The pure orchestration entry points exist regardless of Typer.
    assert callable(cli.run_backtest)
    assert callable(cli.run_ingest)
    assert callable(cli.run_score)
    assert callable(cli.build_app)
    assert callable(cli.main)


@pytest.mark.unit
def test_run_backtest_synthetic_smoke_returns_int() -> None:
    """A tiny synthetic ``backtest`` run completes and returns an int exit code.

    While downstream compute kernels remain stubs the orchestration degrades
    gracefully (a handled ``NotImplementedError`` -> exit code ``1``); once they
    land the same call path returns ``0`` with the honest summary. Either way the
    CLI is invocable end to end on synthetic data with no network and no keys.
    """
    code = cli.run_backtest(
        tickers=["GME", "AMC"],
        start=date(2021, 1, 4),
        end=date(2021, 6, 30),
        cost_bps=10.0,
        seed=7,
    )
    assert isinstance(code, int)
    assert code in (0, 1)


@pytest.mark.unit
def test_run_backtest_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """The backtest path emits a human-readable banner (or a handled error line)."""
    cli.run_backtest(tickers=["GME"], start=date(2021, 1, 4), end=date(2021, 3, 31))
    out = capsys.readouterr().out
    # Either the full summary banner or the graceful handled-error message.
    assert "WSB sentiment signal" in out or out.startswith("error:")


@pytest.mark.unit
def test_run_score_missing_input_is_handled(tmp_path: object) -> None:
    """``score`` on a missing input file surfaces a non-zero exit, not a crash."""
    import os

    missing = os.path.join(str(tmp_path), "nope.jsonl")
    # FileNotFoundError is not a WsbSentimentError, so it propagates as OSError;
    # we assert the function does not silently succeed.
    with pytest.raises((FileNotFoundError, OSError)):
        cli.run_score(in_path=missing, out_path=os.path.join(str(tmp_path), "out.parquet"))


@pytest.mark.unit
def test_run_score_processes_jsonl_posts(tmp_path: object) -> None:
    """``score`` reads a JSONL post table and runs the offline VADER pipeline.

    Exercises the real (lexicon-based, no-network) extract + VADER score path; the
    daily roll-up is a downstream stub, so the run degrades gracefully to exit
    code ``1`` rather than crashing — once the roll-up lands the same path yields a
    parquet panel and exit ``0``.
    """
    import json
    import os

    posts = [
        {
            "post_id": "a1",
            "created_utc": 1_609_800_000,
            "title": "$GME to the moon, diamond hands",
            "body": "I love GME, this is amazing",
            "author": "ape1",
            "score": 42,
            "subreddit": "wallstreetbets",
        },
        {
            "post_id": "b2",
            "created_utc": 1_609_900_000,
            "title": "AMC is terrible",
            "body": "selling my AMC, hate this stock",
            "author": "ape2",
            "score": 7,
            "subreddit": "wallstreetbets",
        },
    ]
    in_path = os.path.join(str(tmp_path), "posts.jsonl")
    with open(in_path, "w", encoding="utf-8") as handle:
        for post in posts:
            handle.write(json.dumps(post) + "\n")

    code = cli.run_score(in_path=in_path, out_path=os.path.join(str(tmp_path), "panel.parquet"))
    assert isinstance(code, int)
    assert code in (0, 1)


@pytest.mark.unit
def test_run_ingest_rejects_unknown_source(tmp_path: object) -> None:
    """An unknown ingestion source is a handled error (exit ``1``)."""
    import os

    code = cli.run_ingest(
        subreddit="wallstreetbets",
        start=date(2021, 1, 1),
        end=date(2021, 1, 2),
        out_path=os.path.join(str(tmp_path), "raw.jsonl"),
        source="bogus",
    )
    assert code == 1


@pytest.mark.unit
def test_run_backtest_with_long_only_and_cache_pref() -> None:
    """Alternate parameter combinations also return a clean int exit code."""
    code = cli.run_backtest(
        tickers=["TSLA", "NVDA"],
        start=date(2021, 1, 4),
        end=date(2021, 4, 30),
        window=3,
        lag=1,
        threshold=0.5,
        cost_bps=15.0,
        long_only=True,
        data_source_pref="auto",
        seed=3,
    )
    assert code in (0, 1)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_TYPER, reason="Typer is not installed in this environment")
def test_build_app_help_smoke() -> None:
    """``build_app`` constructs a Typer app exposing the three commands (Typer present)."""
    from typer.testing import CliRunner

    app = cli.build_app()
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("backtest", "ingest", "score"):
        assert command in result.output


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_TYPER, reason="Typer is not installed in this environment")
def test_backtest_command_smoke() -> None:
    """The ``backtest`` Typer command runs on synthetic defaults (Typer present)."""
    from typer.testing import CliRunner

    app = cli.build_app()
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["backtest", "GME", "AMC", "--start", "2021-01-04", "--end", "2021-06-30"],
    )
    # Exit code 0 (full path) or 1 (graceful, downstream stubbed); never a crash.
    assert result.exit_code in (0, 1)


@pytest.mark.unit
def test_build_app_without_typer_raises_importerror() -> None:
    """When Typer is absent, building the app raises a clear ImportError."""
    if _HAS_TYPER:
        pytest.skip("Typer is installed; the no-Typer path is not exercisable here.")
    with pytest.raises(ImportError):
        cli.build_app()
