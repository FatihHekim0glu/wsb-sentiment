"""End-to-end integration tests for the public ``run_sentiment_backtest`` entry point.

Exercises the full backend path on the SYNTHETIC default: load the sentiment +
price panel -> sweep the window x lag x threshold x cost grid with strict
no-lookahead guards -> select the in-sample-best config -> evaluate it
out-of-sample -> derive the pure verdict. The honest-null discipline requires
``signal_has_edge`` to read ``False`` on the synthetic default (the in-sample edge
decays out-of-sample and fails the DSR + cost hurdles), and the figure helper must
produce the two response figures as plain ``{"data", "layout"}`` dicts.
"""

from __future__ import annotations

from datetime import date

import pytest

from wsb_sentiment import (
    build_sentiment_figures,
    run_sentiment_backtest,
)
from wsb_sentiment._exceptions import InsufficientDataError, ValidationError

_START = date(2021, 1, 4)
_END = date(2022, 12, 30)

#: The flat summary keys the backend response contract relies on.
_REQUIRED_SUMMARY_KEYS = (
    "net_sharpe",
    "buyhold_sharpe",
    "deflated_sharpe",
    "psr",
    "pbo",
    "hac_tstat",
    "hac_pvalue",
    "turnover",
    "n_effective_trials",
    "signal_has_edge",
    "data_source",
)


@pytest.mark.integration
def test_run_sentiment_backtest_synthetic_default_has_no_edge() -> None:
    """The synthetic default runs end-to-end and yields the honest null verdict."""
    run = run_sentiment_backtest(start=_START, end=_END, seed=7)

    # Every contract key is present and the verdict is a bool.
    for key in _REQUIRED_SUMMARY_KEYS:
        assert key in run.summary, f"summary missing {key!r}"
    assert run.summary["data_source"] == "synthetic"
    assert isinstance(run.summary["signal_has_edge"], bool)

    # Honest-null discipline: the in-sample edge decays OOS and fails the hurdles.
    assert run.summary["signal_has_edge"] is False
    assert run.verdict.signal_has_edge is False
    assert run.summary["deflated_sharpe"] is not None
    assert run.summary["deflated_sharpe"] < 0.95

    # The DSR multiplicity is honest: effective trials deflate below the raw grid.
    assert run.n_grid_trials > 1
    assert 1.0 <= run.stats.n_effective_trials <= run.n_grid_trials


@pytest.mark.integration
def test_run_sentiment_backtest_is_deterministic() -> None:
    """A fixed ``(tickers, start, end, seed)`` reproduces an identical summary."""
    first = run_sentiment_backtest(start=_START, end=_END, seed=11)
    second = run_sentiment_backtest(start=_START, end=_END, seed=11)
    assert first.summary == second.summary
    assert first.selected_spec == second.selected_spec


@pytest.mark.integration
def test_run_sentiment_backtest_respects_request_parameters() -> None:
    """Requested tickers and a custom cost flow through and the grid includes them."""
    run = run_sentiment_backtest(
        tickers=["AAA", "BBB", "CCC"],
        start=_START,
        end=_END,
        window=3,
        lag=2,
        threshold=0.25,
        cost_bps=15.0,
        long_only=True,
        seed=3,
    )
    assert run.meta["tickers"] == ["AAA", "BBB", "CCC"]
    # The requested config is always inside the swept grid, so a selection is made.
    assert run.summary["n_grid_trials"] >= 1
    assert run.summary["signal_has_edge"] is False
    # turnover is a finite non-negative scalar (or None on a flat book).
    turnover = run.summary["turnover"]
    assert turnover is None or turnover >= 0.0


@pytest.mark.integration
def test_build_sentiment_figures_shapes() -> None:
    """The figure helper returns the two response figures as plain dicts."""
    run = run_sentiment_backtest(start=_START, end=_END, seed=7)
    figures = build_sentiment_figures(run)

    assert set(figures) == {"equity_figure", "sentiment_figure"}
    for fig in figures.values():
        assert set(fig) == {"data", "layout"}
        assert isinstance(fig["data"], list)
        assert isinstance(fig["layout"], dict)

    # Equity figure plots the signal vs buy-and-hold; sentiment plots both axes.
    assert len(figures["equity_figure"]["data"]) == 2
    assert len(figures["sentiment_figure"]["data"]) == 2

    # A single-ticker sentiment figure is also assembled cleanly.
    single = build_sentiment_figures(run, ticker=run.mean_compound.columns[0])
    assert set(single["sentiment_figure"]) == {"data", "layout"}


@pytest.mark.integration
def test_run_sentiment_backtest_rejects_bad_parameters() -> None:
    """Out-of-range request parameters are rejected before any compute."""
    with pytest.raises(ValidationError, match="window"):
        run_sentiment_backtest(start=_START, end=_END, window=0)
    with pytest.raises(ValidationError, match="lag"):
        run_sentiment_backtest(start=_START, end=_END, lag=0)
    with pytest.raises(ValidationError, match="threshold"):
        run_sentiment_backtest(start=_START, end=_END, threshold=-0.1)
    with pytest.raises(ValidationError, match="cost_bps"):
        run_sentiment_backtest(start=_START, end=_END, cost_bps=-1.0)


@pytest.mark.integration
def test_run_sentiment_backtest_too_short_raises() -> None:
    """A date range too short for a train/test split raises InsufficientDataError."""
    with pytest.raises(InsufficientDataError):
        run_sentiment_backtest(start=date(2021, 1, 4), end=date(2021, 1, 5))
