# wsb-sentiment

Turn r/wallstreetbets chatter into a daily per-ticker sentiment signal and
**honestly** test whether it predicts next-day returns on a point-in-time
S&P 500 universe, using the Deflated Sharpe Ratio, PBO/CSCV, and HAC.

> **Headline (honest weak/negative result).** A naive VADER WSB daily-sentiment
> signal shows a *mild in-sample correlation* with next-day returns that is
> dominated by contemporaneous attention/return feedback and **largely decays
> out-of-sample**, failing the Deflated Sharpe and per-side cost hurdles. The
> verdict reads **`signal_has_edge = False`**, a credible weak/negative result,
> not a profitable edge. Mild in-sample correlation that decays out-of-sample
> after costs is *attention feedback, not alpha*.

The shipped default runs on a **synthetic** sentiment + price generator (no keys),
constructed so the in-sample edge decays out-of-sample and fails the DSR after
costs, the honest null by construction. The real-data path (Pushshift/PRAW +
Polygon point-in-time prices) lives behind the offline `ingest`+`score` CLI.

## Result on the synthetic default

Running the shipped default (`GME, AMC, TSLA, AAPL, NVDA`, 2021-01-04 →
2022-12-30, `window=1, lag=1, threshold=0.0, cost_bps=10, seed=7`) sweeps a
`window × lag × threshold × cost` grid of **54 configurations**, deliberately
selects the *in-sample-best* config, then evaluates it **out-of-sample**:

| Metric | Value | Reads |
| --- | ---: | --- |
| OOS net Sharpe (selected, after cost) | **0.29** | positive, but… |
| Buy-and-hold Sharpe (same OOS window) | 0.94 | the signal underperforms simply holding |
| **Deflated Sharpe Ratio** | **0.26** | well below 0.95, fails the multiplicity hurdle |
| Probabilistic Sharpe Ratio (PSR) | 0.61 | not credible at the full-grid trial count |
| PBO (CSCV) | 0.39 | < 0.5, passes the overfitting hurdle |
| HAC (Newey-West) t-stat | 0.26 | insignificant |
| HAC p-value | 0.79 | well above 0.05, fails the significance hurdle |
| Turnover (avg per-day) | 0.27 | costs bite |
| Effective trials (PCA of grid) | 10.0 | of 54 raw configs |
| **`signal_has_edge`** | **`False`** | the honest null |

The OOS net Sharpe is *nominally* positive, but the after-multiplicity Deflated
Sharpe (0.26) and the HAC test (p = 0.79) both fail, so the pure verdict is
`False`. By construction the in-sample edge is real but **decays out-of-sample**;
what survives is contemporaneous attention/return feedback, not forecasting power.

## Status

The compute core is **implemented** end-to-end: import-pure typed `src/` package,
finance-augmented VADER (+ TextBlob parity), as-of daily roll-up, train-only-scaler
signal, purge/embargo walk-forward, DSR/PSR · PBO/CSCV · HAC · Memmel-JK, and the
pure `signal_has_edge` verdict. Tests: **292 passed**, coverage **about 94%**
(gate 85), `ruff` + `mypy --strict` clean.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'      # lean: NO transformers/torch/TF
# offline ingestion adapters: add the `ingest` extra (praw)
```

```python
import wsb_sentiment   # import-pure: no praw / network / torch at import
```

## Quickstart

The single public entry point runs the whole leakage-guarded pipeline on the
synthetic default (no keys, no live ingest, no VADER scoring at request time):

```python
from datetime import date
from wsb_sentiment import run_sentiment_backtest, build_sentiment_figures

run = run_sentiment_backtest(
    start=date(2021, 1, 4), end=date(2022, 12, 30),
    window=1, lag=1, threshold=0.0, cost_bps=10, seed=7,
)
print(run.summary["signal_has_edge"])     # -> False (the honest null)
figures = build_sentiment_figures(run)     # {"equity_figure", "sentiment_figure"}
```

`run.summary` carries the metrics the backend response exposes: `net_sharpe`,
`buyhold_sharpe`, `deflated_sharpe`, `psr`, `pbo`, `hac_tstat`, `hac_pvalue`,
`turnover`, `n_effective_trials`, `signal_has_edge`, and `data_source`. The
result is deterministic for a fixed `(tickers, start, end, seed)`.

## Reproduce

```bash
# 1. Environment + lean install (no torch/transformers).
uv venv && uv pip install -e '.[data,viz,dev]'

# 2. The exact quality gates CI runs.
uv run ruff check src                                                       # lint
uv run mypy src                                                             # types (strict)
uv run pytest -q --cov=wsb_sentiment --cov-report=term --cov-fail-under=85  # tests + cov

# 3. The headline number, end to end on the synthetic default.
uv run python - <<'PY'
from datetime import date
from wsb_sentiment import run_sentiment_backtest
run = run_sentiment_backtest(start=date(2021, 1, 4), end=date(2022, 12, 30),
                             window=1, lag=1, threshold=0.0, cost_bps=10, seed=7)
for k in ("net_sharpe", "buyhold_sharpe", "deflated_sharpe", "psr", "pbo",
          "hac_tstat", "hac_pvalue", "n_effective_trials", "signal_has_edge"):
    print(f"{k:>20}: {run.summary[k]}")
PY
# -> signal_has_edge: False
```

The synthetic generator is seeded (PCG64 substreams), so the metrics in the table
above reproduce byte-identically for the same `(tickers, start, end, seed)`.

## Design

- **Lexicon sentiment only.** Finance-augmented VADER (primary) with a TextBlob
  cross-check. No transformers/torch/TF; no model fit; no NLTK download at import.
- **Offline ingestion.** Pushshift/PRAW adapters are batch tools behind the
  `[ingest]` extra; they are never called at request time.
- **Leakage guards.** A strict as-of cutoff at the prior session close
  ([ADR-0001](docs/decisions/0001-asof-cutoff-prior-close.md)), `signal.shift(1)`,
  forward-return labels only, `pct_change(fill_method=None)`, a **train-only**
  standardizer ([ADR-0003](docs/decisions/0003-train-only-scaler.md)),
  walk-forward purge + embargo, and a **point-in-time** S&P 500 universe (no
  future-constituent selection,
  [ADR-0002](docs/decisions/0002-pit-universe-no-future-constituent.md)).
- **Honest stats.** Deflated/Probabilistic Sharpe with an *effective* trial count
  (PCA of the swept grid), PBO via CSCV, HAC (Newey-West) t-stats, and the
  Memmel-JK test versus buy-and-hold. The `signal_has_edge` verdict is a **pure
  function** of OOS net Sharpe + DSR + PBO + HAC
  ([ADR-0004](docs/decisions/0004-honest-weak-null.md)).
- **Synthetic-by-default.** The deployed tool reads a precomputed/synthetic daily
  sentiment table and runs only the lightweight backtest at request time
  ([ADR-0005](docs/decisions/0005-synthetic-default-no-live-ingest.md)).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture, data flow, and
invariants, and [`docs/decisions/`](docs/decisions/) for the contested choices.

## Validation

| Suite | What it pins | Tolerance / band |
| --- | --- | --- |
| **parity**, DSR/PSR | PSR-vs-zero, the DSR as PSR vs the expected-maximum benchmark, `n_trials=1` collapse, and the HAC t-stat helper, each re-derived from its closed form and checked against the reused `evaluation.dsr` | **1e-10** |
| **parity**, VADER vs TextBlob | sign agreement on a labelled polar corpus (positive / negative); neutral text near zero; finance-lexicon boosters move bullish/bearish jargon in the correct sign; probabilities sum to 1 | **unanimous sign agreement** on the polar fixtures; neutral `\|compound\|` near 0 (abs 1e-9); `pos+neu+neg = 1` (abs 1e-6) |
| **parity**, roll-up | the as-of daily aggregation matches a slow Python reference roll-up element-by-element | exact (slow oracle) |
| **property** (Hypothesis) | as-of prefix-determinism / future-perturbation invariance of the aggregator; `signal.shift` equivariance; standardization scale-invariance; mention-count monotonicity | invariants hold for all generated inputs |
| **regression** | golden decaying-signal backtest (in-sample edge → OOS decay → `signal_has_edge = False` after DSR/costs); the no-lookahead golden test; the import-purity subprocess test | locked metrics + `False` verdict |
| **integration** | end-to-end score → aggregate → signal → backtest on synthetic data | runs green, deterministic |

The DSR/PSR primitives are validated **oracle → test**: an independent closed-form
oracle is derived in the test from the published formulae, then the library output
is asserted equal to it to 1e-10, so the reused `dsr.py` is pinned to the
literature, not merely to itself.

## Limitations

- **Pushshift deletion bias + coverage gap.** The Reddit/Pushshift historical
  archive is missing removed/deleted posts and its coverage thins markedly after
  the 2023 API changes, biasing any reconstructed historical sentiment downward
  and non-randomly (survivorship in *posts*, not just tickers).
- **PIT survivorship.** Only symbols in the S&P 500 universe **as-of each date**
  are tradable; meme tickers outside the as-of universe are **descriptive-only**
  and never traded. The tool reports their sentiment but does not take positions
  in them, precisely to avoid future-constituent / survivorship selection.
- **Meme tickers are descriptive-only.** The most WSB-discussed names (e.g. GME,
  AMC) are shown for context but are typically outside the PIT-tradable book; their
  chatter informs the figures, not the traded signal.
- **Synthetic default.** The shipped tool runs on a synthetic generator so the
  result is reproducible and the null is honest; real data flows through the
  offline `ingest` + `score` path and a Polygon PIT-price provider, which are not
  exercised by the deployed request path.

## References

- DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification*. The
  out-of-sample decay of estimated strategies versus a naive benchmark; the canonical
  OOS-decay precedent.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*. DSR/PSR correcting
  for selection bias, multiplicity, and non-normality.
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
  Overfitting*. PBO via Combinatorially Symmetric Cross-Validation (CSCV).
- Hutto & Gilbert (2014), *VADER: A Parsimonious Rule-Based Model for Sentiment
  Analysis of Social Media Text*. The lexicon-based scorer used here.
- Newey & West (1987); Andrews (1991). Heteroskedasticity- and
  autocorrelation-consistent (HAC) standard errors.

## License

MIT, see [LICENSE](LICENSE).
</content>
</invoke>
