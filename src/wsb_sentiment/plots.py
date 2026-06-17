"""Plotly figure builders (LAZY plotly).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so no Plotly object leaks across the API boundary. Plotly is
an OPTIONAL dependency (the ``viz`` extra) imported LAZILY inside each builder;
importing this module has no side effects and does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout} shape).


def oos_equity_figure(
    net_returns: pd.Series,
    buyhold_returns: pd.Series,
    *,
    title: str = "Out-of-sample equity: signal vs buy-and-hold",
) -> FigureDict:
    """Build the OOS equity-curve figure (signal vs buy-and-hold).

    Plots cumulative wealth for the net (after-cost) sentiment signal against the
    equal-weight buy-and-hold baseline over the out-of-sample window.

    Parameters
    ----------
    net_returns:
        The net OOS return series of the signal.
    buyhold_returns:
        The OOS return series of the buy-and-hold baseline.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the two equity curves.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("oos_equity_figure is not yet implemented")


def sentiment_figure(
    mean_compound: pd.DataFrame,
    mention_count: pd.DataFrame,
    *,
    ticker: str | None = None,
    title: str = "Daily WSB sentiment and mention count",
) -> FigureDict:
    """Build the daily-sentiment + mention-count figure.

    Plots the daily mean compound sentiment (left axis) against the mention count
    (right axis), either aggregated or for a single ``ticker``.

    Parameters
    ----------
    mean_compound:
        Wide ``day x ticker`` mean compound sentiment.
    mention_count:
        Wide ``day x ticker`` mention count.
    ticker:
        Optional single ticker to plot; when ``None`` an aggregate is shown.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering sentiment and mention count.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("sentiment_figure is not yet implemented")
