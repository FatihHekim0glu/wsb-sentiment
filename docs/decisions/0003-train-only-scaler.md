# ADR-0003: Standardizer fit on the train slice only

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** wsb-sentiment maintainers
- **Related:** [ADR-0001](0001-asof-cutoff-prior-close.md) (as-of cutoff),
  [ADR-0004](0004-honest-weak-null.md) (the verdict the guard protects)

## Context

The daily sentiment aggregate is standardized (per-ticker mean/std) before it is
mapped to a position via a score threshold. Standardization is a learned transform:
its mean and standard deviation are *parameters estimated from data*. If those
parameters are estimated over the **whole** sample — including the out-of-sample
window — then the OOS positions are computed using statistics that "saw" the OOS
distribution. That is a subtle but real look-ahead: the activation threshold is
calibrated against a spread the model is not supposed to know yet, and it
systematically flatters OOS performance.

This is the same family of bug as fitting a `StandardScaler` on the full dataset
before a train/test split — a textbook leak that is easy to commit by accident in a
sweep that re-uses one scaler across configurations.

## Decision

The standardizer is **fit on the TRAIN slice only**. In `signal/build.py`,
`fit_standardizer(sentiment, train_end, window)` computes the per-ticker mean and
std over observations with index `<= train_end` (std floored at `EPS`), records
`n_train`, and returns a frozen `StandardizerState`. `build_positions` then *applies*
those train-fit parameters to the full series, so the OOS standardized scores use
only train-era statistics.

In the sweep (`backtest/runner.py`) a fresh standardizer is fit on the train slice
for **every** configuration, so no OOS information ever enters parameter estimation
for any trial.

## Consequences

- **Positive.** The OOS standardized signal is computed with no knowledge of the OOS
  distribution; the leak is closed at the only place it could enter.
- **Positive.** The guard is an *invariant*: perturbing OOS sentiment leaves the
  fitted `(mean, std)` — and therefore the train-era positions — unchanged
  (future-perturbation invariance, property-tested). Standardization is also scale-
  invariant by construction (a positive affine rescale of the raw sentiment leaves
  the standardized signal unchanged), which is separately property-tested.
- **Cost.** If the train slice is short or regime-shifted, the train-fit mean/std can
  be a poor description of the OOS spread, which can *hurt* OOS performance — but
  that is the honest cost of not peeking, and it is part of why the edge decays.
- **Risk addressed.** "Fit the scaler on the full sample" is rejected; the scaler is
  train-only and tested.
</content>
