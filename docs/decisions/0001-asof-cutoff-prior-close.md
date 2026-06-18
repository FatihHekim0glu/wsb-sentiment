# ADR-0001: As-of cutoff at the prior session close

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** wsb-sentiment maintainers
- **Related:** [ADR-0003](0003-train-only-scaler.md) (train-only standardizer),
  [ADR-0004](0004-honest-weak-null.md) (the pure verdict the guard protects)

## Context

Reddit posts arrive continuously, with a `created_utc` timestamp. A daily
sentiment signal must decide *which trading day each post is allowed to inform*.
The tempting, and wrong, choice is to bucket posts by calendar date and use a
day's full-day sentiment to predict that **same** day's return. That is
look-ahead: a post made at 15:55, or worse at 21:00 after the close, would be
folded into a "signal" credited with predicting a move that already happened (or
that the post is reacting to). Sentiment and contemporaneous returns are
mechanically entangled on r/wallstreetbets, attention spikes *after* a move,
so any same-bar leakage manufactures a spurious edge.

This is the single largest correctness risk in the whole pipeline. It must be
closed at the aggregation boundary, before any standardization or backtest, and
it must be enforced by tests, not by convention.

## Decision

The roll-up uses a **strict as-of cutoff at the prior session close**. In
`aggregate/rollup.py`:

- Each post's `created_utc` is interpreted in the exchange timezone.
- A post created **after** a session's close (default `16:00` exchange-local)
  rolls into the **next** trading session's signal, not the current one.
- The daily per-(ticker, day) aggregate for day *D* is therefore built only from
  posts available **before day *D*'s open**.

Downstream, `signal/build.py` applies positions via `signal.shift(lag)` (default
`lag = 1`) and labels are **forward returns only** (`pct_change(fill_method=None)`,
never same-bar). The cutoff and the shift are complementary: the cutoff fixes
*which information* a day's signal may contain; the shift fixes *which return* that
signal is allowed to trade.

## Consequences

- **Positive.** Same-bar contamination, the dominant source of a fake
  sentiment→return correlation, is structurally impossible. The mild in-sample
  correlation that survives is honest, and it is exactly what decays out-of-sample.
- **Positive.** The guard is testable as an *invariant*: appending or perturbing
  future posts leaves every already-emitted daily aggregate unchanged
  (prefix-determinism / future-perturbation invariance, property-tested).
- **Cost.** One session of signal is "lost" at each edge (the most recent posts do
  not trade until the next open), and the session-close time / timezone must match
  the exchange. These are stated explicitly so they are not silently changed.
- **Risk addressed.** "Bucket by calendar day and predict the same day" is
  rejected; the cutoff is justified by the data-generating structure and tested.
</content>
