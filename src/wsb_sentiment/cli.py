"""Command-line interface (Typer).

A thin orchestration layer over the compute library exposing the OFFLINE
``ingest`` / ``score`` path and the ``backtest`` path:

- ``ingest`` — pull raw WSB posts (Pushshift/PRAW) into a local table (offline);
- ``score`` — VADER(+TextBlob parity) score the ingested text and roll up to a
  daily per-ticker sentiment panel (offline);
- ``backtest`` — run the leakage-guarded sentiment-signal backtest and print the
  honest summary + verdict (defaults to the synthetic generator).

Constructing the app object is deferred to :func:`build_app` so importing this
module has no side effects (no command registration, no Typer import, no I/O at
import time). The module-level ``app`` is a lazily-built singleton consumed by the
``wsb-sentiment`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the ``ingest``, ``score``, and ``backtest`` commands on a fresh
    ``typer.Typer`` instance. Typer is imported LAZILY inside this function so
    that importing :mod:`wsb_sentiment.cli` imports no Typer and registers no
    commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("build_app is not yet implemented")


def main() -> None:
    """Console-script entry point: build the app and dispatch.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("main is not yet implemented")
