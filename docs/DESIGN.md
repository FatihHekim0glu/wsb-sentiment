# Design

This document explains how `wsb-sentiment` is put together: the layering, the
data flow from a scored post to the pure edge verdict, the leakage invariants the
compute core guarantees, and the testing strategy that keeps the honest headline
honest. For *why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI,
  network, praw, or model dependencies along.
- A faithful lexicon sentiment pipeline (finance-augmented VADER, primary; TextBlob
  as a parity cross-check), no transformers/torch/TF, no model fit, no NLTK
  download at import.
- A statistically defensible verdict that survives multiplicity correction and is
  *mechanically* prevented from over-claiming.
- A reproducible **honest null**: on the synthetic default the in-sample edge
  decays out-of-sample and fails the Deflated Sharpe after costs.

**Non-goals**

- Profiting from WSB chatter. The honest finding is that the naive daily-sentiment
  signal does *not*, after costs and multiplicity, beat buy-and-hold by a
  significant margin.
- A live trading or live-ingestion system. The deployed path runs a lightweight
  backtest on a precomputed/synthetic sentiment table; Pushshift/PRAW scoring is an
  offline batch tool ([ADR-0005](decisions/0005-synthetic-default-no-live-ingest.md)).
- A general NLP toolkit. The scorers exist only to produce a daily sentiment panel.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/wsb_sentiment/` has **zero import-time side effects**, guarded by a subprocess
import-purity test. Heavy dependencies (praw, vaderSentiment, textblob, plotly,
typer) are imported lazily inside the functions that use them.

```
                 cli.py (Typer)        plots.py (Plotly)
                      |                      |
   ┌──────────────────┴──────────────────────┴───────────────────────┐
   │                          backtest/runner.py                       │
   │       run_sentiment_backtest  ·  build_sentiment_figures          │
   │   (the single public entry point the FastAPI router calls)        │
   ├───────────────────────────────────────────────────────────────────
   │                          evaluation/                              │
   │   dsr.py · stats.py · pbo.py · cpcv.py · hac.py · memmel.py        │
   │   bootstrap_ci.py · verdict.py (pure signal_has_edge deriver)      │
   ├───────────────────────────────────────────────────────────────────
   │                          backtest/                                │
   │   engine.py · walk_forward.py · costs.py · stats.py               │
   │   (no-lookahead engine · purge/embargo · per-side bps · Sharpe)    │
   ├───────────────────────────────────────────────────────────────────
   │   signal/build.py                       aggregate/rollup.py        │
   │   train-only standardizer +             as-of cutoff at prior      │
   │   shift(lag) positions                  close → daily panel        │
   ├───────────────────────────────────────────────────────────────────
   │   nlp/                                   ingest/                    │
   │   vader.py · textblob_parity.py         pushshift.py · reddit_api  │
   │   (lexicon scoring, no fit)             extract.py (OFFLINE batch)  │
   ├───────────────────────────────────────────────────────────────────
   │   data.py  (synthetic sentiment+price generator, loaders,         │
   │             PIT-universe hook, compute_returns)                    │
   ├───────────────────────────────────────────────────────────────────
   │   foundation (no internal deps):  _validation · _constants         │
   │   _typing · _exceptions · _manifest · _rng                         │
   └───────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py`, `TRADING_DAYS` / `PERIODS_PER_YEAR` / `EPS`; one source of truth.
- `_validation.py`, input guards (shape, finiteness, sufficient observations,
  inner alignment).
- `_typing.py` / `_exceptions.py`, shared aliases and the `WsbSentimentError`
  exception taxonomy (every vendored primitive is re-homed onto this base).
- `_manifest.py` / `_rng.py`, `RunManifest` (BLAKE2b config hash) plus seeded
  PCG64 substreams. The same seed yields byte-identical sentiment, prices, metrics,
  and verdict.

### `ingest/` and `nlp/`

`ingest/` holds the **offline batch** Reddit/Pushshift adapters (lazy clients,
never imported or called at request time) and `extract.py` ($TICKER / cashtag +
bare-symbol mention extraction with de-duplication, retaining `created_utc`).
`nlp/` holds the lexicon scorers: `vader.py` (finance-lexicon-augmented VADER,
primary, static lexicon, no model fit) and `textblob_parity.py` (a TextBlob
cross-check). Both score per post and are deterministic.

### `aggregate/rollup.py`

Aggregates scored mentions to a per-(ticker, day) panel, mean/median compound,
mention count, positive-share, under a **strict as-of cutoff at the prior session
close** ([ADR-0001](decisions/0001-asof-cutoff-prior-close.md)). A post created
after a session's close rolls into the *next* session's signal, so a day's signal
can only ever be informed by information available before that day's open.

### `signal/build.py`

Standardizes the daily aggregate with a scaler whose mean/std are fit on the
**train slice only** ([ADR-0003](decisions/0003-train-only-scaler.md)), then maps
the standardized score to a long/short (or long/flat) position. Positions are
applied via `signal.shift(lag)`, so today's signal trades tomorrow.

### `backtest/`

`engine.py` builds each configuration's net-return stream on a shared
post-purge/embargo OOS index, against buy-and-hold and an attention-only
(mention-count) baseline, applying the PIT `universe_mask`. `walk_forward.py` is the
anchored no-lookahead engine (purge + embargo, `signal.shift`). `costs.py` applies
the per-side bps grid against realized turnover. `stats.py` holds Sharpe, annualized
vol, max-drawdown, and turnover. `runner.py` orchestrates the whole pipeline and is
the single public entry point.

### `evaluation/`

`dsr.py` computes the Deflated/Probabilistic Sharpe with the full-grid effective
`n_trials`. `stats.py` assembles the `HonestStats` bundle, DSR/PSR with a
**PCA-of-trial-returns effective trial count** over the swept grid, PBO via CSCV
(`pbo.py` / `cpcv.py`), HAC (Newey-West) t-stats (`hac.py`, Andrews bandwidth), the
Memmel-JK test versus buy-and-hold (`memmel.py`), and stationary-bootstrap CIs
(`bootstrap_ci.py`). `verdict.py` is a **pure function** mapping
`(oos_net_sharpe, deflated_sharpe, pbo, hac_pvalue)` to `signal_has_edge`.

## Data flow: post → verdict

```
scored posts (compound, created_utc, ticker)
        │  as-of cutoff @ prior close  (aggregate/rollup.py)
        ▼
daily per-ticker panel (mean compound, mention count, positive-share)
        │  + synthetic/cache prices ─► compute_returns  (pct_change(fill_method=None))
        ▼
anchored train/test split at the sample midpoint
        │  SWEEP window × lag × threshold × cost  (54 configs by default)
        │     each config: fit standardizer on TRAIN only ─► build_positions
        │                  ─► shift(lag) ─► engine: net stream on shared OOS index
        ▼
deliberately SELECT the in-sample-best config  (the selection bias the DSR/PBO penalise)
        │
        ▼
HonestStats over the FULL grid:
   net/buy-hold Sharpe · DSR/PSR (PCA-effective n_trials) · PBO/CSCV · HAC t-stat · Memmel-JK
        │
        ▼
verdict.derive_verdict  ──►  signal_has_edge  (pure-derived bool)
```

The synthetic generator is built so the *selected* in-sample-best config has a real
in-sample edge that **decays out-of-sample**; the DSR (0.26 « 0.95) and the HAC test
(p = 0.79) then fail, so `signal_has_edge` reads `False`.

## Key invariants

The compute core guarantees, and tests enforce:

1. **As-of cutoff.** A day's roll-up depends only on posts created before that
   day's open (prior-close cutoff). Future posts cannot leak into a past day.
2. **Prefix-determinism / future-perturbation invariance.** Perturbing or
   appending future posts leaves every already-emitted daily aggregate unchanged.
3. **No-lookahead.** Standardizer mean/std are fit on the TRAIN slice only;
   perturbing OOS data does not change them. Positions are `signal.shift(lag)`-ed
   and labels are forward returns only, never same-bar.
4. **Standardization scale-invariance.** A positive affine rescale of the raw
   sentiment leaves the standardized signal (hence the positions) unchanged.
5. **Mention-count monotonicity.** The attention-only baseline is monotone in the
   mention count it is built from.
6. **PIT universe.** The traded book is restricted to symbols in the S&P 500
   universe as-of each date; meme tickers outside it are descriptive-only
   ([ADR-0002](decisions/0002-pit-universe-no-future-constituent.md)).
7. **DSR multiplicity.** The Deflated Sharpe is deflated by the PCA-effective trial
   count over the full swept grid; it is non-increasing in `n_trials`.
8. **Verdict safety.** `signal_has_edge` cannot read `True` unless OOS net Sharpe
   > 0 AND DSR > threshold AND PBO < threshold AND HAC p < α, all at once
   (truth-table unit-tested, [ADR-0004](decisions/0004-honest-weak-null.md)).
9. **Determinism.** Same `(tickers, start, end, seed)` → byte-identical outputs.
10. **Import purity.** Importing any `src/wsb_sentiment` module triggers no I/O, no
    network, no praw/vader/textblob/plotly/typer import (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`):

- **`unit/`**, isolated kernels: the verdict truth table, the standardizer, the
  CLI surface, individual evaluation primitives.
- **`property/`** (Hypothesis), the invariants above: as-of prefix-determinism,
  shift-equivariance, standardization scale-invariance, mention-count monotonicity.
- **`parity/`**, golden checks against independent references: VADER↔TextBlob sign
  agreement on a labelled polar corpus; the daily roll-up vs a slow Python reference;
  DSR/PSR and the HAC t-stat helper vs independently re-derived closed forms to
  **1e-10**.
- **`regression/`**, the honest null, locked: the decaying-signal golden backtest
  (in-sample edge → OOS decay → `signal_has_edge = False` after DSR/costs), the
  no-lookahead golden test, and the import-purity subprocess test.
- **`integration/`**, end-to-end score → aggregate → signal → backtest on the
  synthetic panel.

Seeded fixtures in `conftest.py` (`synthetic_sentiment_panel`, `decaying_signal`,
`pure_noise`) give every layer deterministic, adversarial inputs.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`wsb-sentiment[data]` (lean, **no** torch/transformers) under
`api/lib/wsb_sentiment/` and exposes `POST /tools/wsb-sentiment-signal/run`,
returning summary scalars plus Plotly `{data, layout}` figures (the OOS equity curve
signal-vs-buy-hold and the daily sentiment + mention-count chart). The deployed path
**reads a precomputed/synthetic daily sentiment table**, no live Pushshift/PRAW and
no VADER scoring at request time ([ADR-0005](decisions/0005-synthetic-default-no-live-ingest.md)).
The frontend renders the figures and surfaces the pure-derived `signal_has_edge` as a
prominent **"Signal has edge: NO"** badge, with the honest caption "mild in-sample
correlation that decays out-of-sample after costs, attention feedback, not alpha".
</content>
