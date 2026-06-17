"""Unit tests for the LAZY-plotly figure builders.

Asserts each builder returns a JSON-serializable ``{"data", "layout"}`` mapping
with no Plotly object leaking across the boundary, and that building a figure does
not import Plotly at all (the figures are plain dicts).
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.data import generate_synthetic_panel
from wsb_sentiment.plots import oos_equity_figure, sentiment_figure


def _assert_valid_figure(fig: object) -> None:
    """A figure must be a ``{"data": [...], "layout": {...}}`` JSON-safe mapping."""
    assert isinstance(fig, dict)
    assert set(fig.keys()) == {"data", "layout"}
    assert isinstance(fig["data"], list) and len(fig["data"]) >= 1
    assert isinstance(fig["layout"], dict)
    for trace in fig["data"]:
        assert "type" in trace
    # The whole figure must round-trip through JSON (no numpy/plotly objects).
    json.dumps(fig)


@pytest.fixture
def _returns_pair() -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2022-01-03", periods=40, freq="B")
    gen = np.random.default_rng(0)
    net = pd.Series(gen.normal(0.0002, 0.01, size=40), index=index)
    buyhold = pd.Series(gen.normal(0.0003, 0.012, size=40), index=index)
    return net, buyhold


@pytest.mark.unit
def test_oos_equity_figure_is_valid(_returns_pair: tuple[pd.Series, pd.Series]) -> None:
    """The equity figure renders two named cumulative-wealth curves."""
    net, buyhold = _returns_pair
    fig = oos_equity_figure(net, buyhold)
    _assert_valid_figure(fig)
    assert len(fig["data"]) == 2
    names = {trace["name"] for trace in fig["data"]}
    assert names == {"Sentiment signal (net)", "Buy & hold"}
    # Cumulative wealth starts near 1 (growth of 1), not at the raw return level.
    first_y = fig["data"][0]["y"][0]
    assert 0.5 < first_y < 1.5


@pytest.mark.unit
def test_oos_equity_figure_handles_nan_returns() -> None:
    """NaN returns are treated as a flat (zero) day, not propagated into wealth."""
    index = pd.date_range("2022-01-03", periods=5, freq="B")
    net = pd.Series([0.01, np.nan, 0.0, -0.01, 0.02], index=index)
    buyhold = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=index)
    fig = oos_equity_figure(net, buyhold)
    _assert_valid_figure(fig)
    assert all(np.isfinite(y) for y in fig["data"][0]["y"])


@pytest.mark.unit
def test_sentiment_figure_aggregate_is_valid() -> None:
    """The aggregate sentiment figure shows a sentiment line and a mention-bar."""
    panel = generate_synthetic_panel(
        ["GME", "AMC", "TSLA"], date(2022, 1, 3), date(2022, 3, 31), seed=1
    )
    fig = sentiment_figure(panel.mean_compound, panel.mention_count)
    _assert_valid_figure(fig)
    types = {trace["type"] for trace in fig["data"]}
    assert types == {"bar", "scatter"}
    assert "aggregate" in fig["layout"]["title"]["text"]
    # The sentiment axis is pinned to the score range.
    assert fig["layout"]["yaxis"]["range"] == [-1.0, 1.0]
    # The mention bar is plotted on the secondary axis.
    bar = next(t for t in fig["data"] if t["type"] == "bar")
    assert bar["yaxis"] == "y2"


@pytest.mark.unit
def test_sentiment_figure_single_ticker() -> None:
    """A single-ticker figure names the ticker in the title."""
    panel = generate_synthetic_panel(["GME", "AMC"], date(2022, 1, 3), date(2022, 2, 28), seed=2)
    fig = sentiment_figure(panel.mean_compound, panel.mention_count, ticker="GME")
    _assert_valid_figure(fig)
    assert "GME" in fig["layout"]["title"]["text"]


@pytest.mark.unit
def test_sentiment_figure_unknown_ticker_raises() -> None:
    """Requesting a ticker absent from the panel raises :class:`ValidationError`."""
    panel = generate_synthetic_panel(["GME"], date(2022, 1, 3), date(2022, 2, 28), seed=2)
    with pytest.raises(ValidationError, match="not in panel columns"):
        sentiment_figure(panel.mean_compound, panel.mention_count, ticker="ZZZZ")


@pytest.mark.unit
def test_iso_index_stringifies_non_datetime_labels() -> None:
    """A non-datetime index (e.g. integer/string labels) is rendered as strings."""
    from wsb_sentiment.plots import _iso_index

    out = _iso_index(pd.Index([0, 1, 2]))
    assert out == ["0", "1", "2"]
    out2 = _iso_index(pd.Index(["GME", "AMC"]))
    assert out2 == ["GME", "AMC"]


@pytest.mark.unit
def test_jsonify_coerces_numpy_and_pandas_scalars() -> None:
    """``_jsonify`` recursively coerces numpy/pandas types to native Python."""
    from wsb_sentiment.plots import _jsonify

    out = _jsonify(
        {
            "arr": np.array([1, 2, 3]),
            "nested": [np.float64(1.5), (np.int64(2), "x")],
            "scalar": np.int32(7),
            "ts": pd.Timestamp("2022-01-03"),
            "period": pd.Period("2022-01", freq="M"),
            "plain": "ok",
        }
    )
    assert out["arr"] == [1, 2, 3]
    assert out["nested"] == [1.5, [2, "x"]]
    assert out["scalar"] == 7 and isinstance(out["scalar"], int)
    assert out["ts"].startswith("2022-01-03")
    assert "2022-01" in out["period"]
    assert out["plain"] == "ok"
    # The result must be JSON-serializable.
    json.dumps(out)


@pytest.mark.unit
def test_building_figures_does_not_import_plotly() -> None:
    """The dict-shaped figures must not pull Plotly into ``sys.modules``."""
    sys.modules.pop("plotly", None)
    panel = generate_synthetic_panel(["GME"], date(2022, 1, 3), date(2022, 1, 31), seed=0)
    sentiment_figure(panel.mean_compound, panel.mention_count)
    oos_equity_figure(
        panel.prices.pct_change(fill_method=None).iloc[:, 0].fillna(0.0),
        panel.prices.pct_change(fill_method=None).iloc[:, 0].fillna(0.0),
    )
    assert "plotly" not in sys.modules
