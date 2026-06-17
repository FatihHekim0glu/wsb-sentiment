# ADR-0005: Synthetic default, no live ingestion at request time

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** wsb-sentiment maintainers
- **Related:** [ADR-0002](0002-pit-universe-no-future-constituent.md) (PIT universe),
  [ADR-0004](0004-honest-weak-null.md) (the honest null this default makes reproducible)

## Context

A tool that "turns r/wallstreetbets chatter into a signal" sounds like it should
hit Reddit live. Doing so at request time is the wrong design on every axis:

- **Reproducibility.** Live Pushshift/PRAW responses change minute to minute;
  nobody could reproduce a reported metric, and the honest null would not be
  *demonstrably* honest.
- **Correctness.** VADER scoring at request time invites the as-of cutoff
  ([ADR-0001](0001-asof-cutoff-prior-close.md)) to be fudged under latency pressure.
- **Cost & footprint.** Scoring thousands of posts per request, and shipping praw +
  a scorer into a small API container, is heavy and fragile; the lean `[data]`
  extra deliberately excludes praw/transformers/torch.
- **Availability.** There are no Reddit/Pushshift/Polygon keys in the deployment
  environment, and the Pushshift archive has a documented deletion bias and a
  post-2023 coverage gap.

We want a default that is reproducible, fast, key-free, and that *builds the honest
null in by construction* — while still keeping the real-data path available and
tested.

## Decision

The shipped default runs on a **synthetic sentiment + price generator** and the
deployed request path performs **no live ingestion and no VADER scoring**:

- `data.py` generates, from a seeded PCG64 stream, a per-(ticker, day) sentiment
  panel (mean compound, mention count, positive-share) plus a correlated-ish price
  panel and a PIT universe mask, constructed so the in-sample sentiment→return
  correlation **decays out-of-sample and fails the DSR after costs** — the honest
  null by construction.
- `run_sentiment_backtest(..., data_source_pref="synthetic")` (the deployed default)
  reads this synthetic table — or a precomputed cached parquet when present
  (`data_source ∈ {synthetic, cache, polygon}`) — and runs **only the lightweight
  backtest** at request time.
- VADER/TextBlob scoring and the Pushshift/PRAW adapters live in the library
  (`nlp/`, `ingest/`), tested on synthetic text fixtures, and are exercised only by
  the **offline** `ingest` + `score` CLI path (behind the `[ingest]` extra). They
  are never imported or called at request time.

## Consequences

- **Positive.** Every reported metric is reproducible from `(tickers, start, end,
  seed)` with no keys and no network; the honest null is demonstrable, not asserted.
- **Positive.** The API container stays lean (no praw/transformers/torch) and fast
  (no per-request scoring), and the as-of cutoff cannot be compromised at request
  time because there is nothing to score.
- **Positive.** The real-data capability is preserved and unit-tested; switching to
  cached/Polygon data is a `data_source` change, not a rewrite.
- **Cost.** The deployed default does not reflect *live* Reddit sentiment — it is a
  faithful synthetic null, clearly labelled as such in the README and the frontend
  `data_source` badge. Real conclusions require running the offline ingest path on
  actual (deletion-biased, coverage-gapped) Pushshift data.
- **Risk addressed.** "Scrape Reddit and score posts on every request" is rejected;
  ingestion is offline batch and the deployed default is reproducible-synthetic.
</content>
