# ADR-0002: Point-in-time universe, no future-constituent selection

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** wsb-sentiment maintainers
- **Related:** [ADR-0005](0005-synthetic-default-no-live-ingest.md) (synthetic default
  simulates a PIT-consistent set)

## Context

r/wallstreetbets discussion is dominated by a handful of meme tickers (GME, AMC,
and rotating names). Two survivorship traps lurk here:

1. **Future-constituent selection.** Building the tradable universe from *today's*
   S&P 500 membership and back-applying it leaks the future: a name is "tradable"
   in 2021 only because we know it is in the index in 2026. This is the classic
   survivorship bias and it flatters any backtest.
2. **Meme-ticker survivorship.** The most-discussed names are precisely the ones
   that mooned or blew up. Trading them because they are loud, without anchoring
   to an objective, point-in-time tradability rule, bakes the outcome into the
   selection.

We need a rule that decides tradability using only information available *as of
each date*, and that does not let the chatter pick the book.

## Decision

The tradable signal is restricted to symbols in the **S&P 500 universe as-of each
date**, a point-in-time membership mask, no future-constituent selection.
`data.pit_universe(...)` returns a wide `day × ticker` boolean mask built only from
the calendar and historical membership:

- The **real** path consults a Polygon S&P-500 point-in-time universe provider.
- The **synthetic** default simulates a PIT-consistent membership set, seeded so
  the mask is deterministic and reproducible.

Symbols **outside** the as-of universe, including most meme tickers on most days,
are **descriptive-only**: their sentiment and mention counts are reported in the
figures, but the backtest never takes a position in them (`universe_mask = False`).
The membership decision is a function of the date alone, never of any future
price or sentiment realization.

## Consequences

- **Positive.** The backtest cannot benefit from knowing which names later joined
  the index, nor from cherry-picking the loud winners; the universe is honest.
- **Positive.** The "meme ticker" use case is still served, descriptively, so the
  tool is useful without being misleading; the frontend shows GME/AMC sentiment
  while clearly not trading them.
- **Cost.** The most WSB-relevant tickers are frequently *not* in the tradable book,
  which is one reason the daily-sentiment signal has little to trade and weakens
  further after costs. This is a faithful feature of the problem, not a bug.
- **Risk addressed.** Both "use today's index members historically" and "trade the
  loudest tickers" are rejected; tradability is PIT and tested via a
  new-constituent / exclusion fixture.
</content>
