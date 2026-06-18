# ADR-0004: An honest weak/negative null, mechanically enforced

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** wsb-sentiment maintainers
- **Related:** [ADR-0001](0001-asof-cutoff-prior-close.md),
  [ADR-0003](0003-train-only-scaler.md) (the leakage guards this verdict depends on),
  [ADR-0005](0005-synthetic-default-no-live-ingest.md)

## Context

"WSB sentiment predicts returns" is exactly the kind of claim that is trivial to
*appear* to support and very hard to support honestly. A sweep over
`window × lag × threshold × cost` will, with near-certainty, surface at least one
configuration with a positive in-sample Sharpe. Reporting that number as "the
result" is selection bias dressed as discovery. The intellectually honest outcome
here is a **weak/negative** one: a mild in-sample correlation, dominated by
contemporaneous attention/return feedback, that **largely decays out-of-sample** and
fails the after-cost, after-multiplicity hurdles.

The danger is not getting the statistics wrong, it is *narrating* a verdict. A
human (or a future contributor) eyeballing a 0.29 OOS Sharpe might be tempted to
write "shows promise". The verdict must be impossible to inflate.

## Decision

The headline `signal_has_edge` is a **pure function** of the honest-statistics
outputs and nothing else. `evaluation/verdict.derive_verdict(...)` returns `True`
**iff all four hurdles pass simultaneously**:

1. `oos_net_sharpe > 0`, the after-cost OOS Sharpe is positive;
2. `deflated_sharpe > dsr_threshold` (default `0.95`), the multiplicity-adjusted
   Sharpe is credible at the **PCA-effective trial count over the full grid**;
3. `pbo < pbo_threshold` (default `0.5`), low probability of backtest overfitting
   (CSCV);
4. `hac_pvalue < alpha` (default `0.05`), the net mean return is HAC-significant.

If any hurdle fails, the verdict is `False`, regardless of the raw Sharpe point
estimate. The function validates its inputs (DSR, PBO, p-value in `[0, 1]`, all
finite) and the truth table is unit-tested. The verdict is **derived, never
narrated**; no prose path can set it.

On the synthetic default this yields `False`: OOS net Sharpe 0.29 (pass) but DSR
0.26 « 0.95 (fail) and HAC p 0.79 » 0.05 (fail), two independent hurdles veto the
edge claim.

## Consequences

- **Positive.** Over-claiming is *mechanically* prevented. The verdict cannot read
  `True` while the DSR fails, the PBO is high, or the HAC test is insignificant.
- **Positive.** The result is robust to the sweep: deliberately selecting the
  in-sample-best config is *expected*, and the DSR/PBO are precisely the penalties
  for that selection. The honesty lives in the deflation, not in pretending we did
  not sweep.
- **Positive.** The frontend can surface a blunt "Signal has edge: NO" badge with
  confidence, because the flag is a deterministic consequence of the evidence.
- **Cost.** A genuinely strong real-data signal would have to clear a demanding bar
  (DSR > 0.95 at the full effective-trials count). That is intentional: the cost of
  not over-claiming is occasionally under-claiming a marginal true effect.
- **Risk addressed.** "Report the best in-sample config as the result" and "narrate
  an optimistic verdict" are both rejected; the verdict is pure and tested.
</content>
