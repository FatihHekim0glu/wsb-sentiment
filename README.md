# wsb-sentiment

Turn r/wallstreetbets chatter into a daily per-ticker sentiment signal and
**honestly** test whether it predicts next-day returns on a point-in-time
S&P 500 universe — with the Deflated Sharpe Ratio, PBO/CSCV, and HAC.

> **Headline (honest weak/negative result).** A naive VADER WSB daily-sentiment
> signal shows a *mild in-sample correlation* with next-day returns that is
> dominated by contemporaneous attention/return feedback and **largely decays
> out-of-sample**, failing the Deflated Sharpe and per-side cost hurdles. The
> verdict reads **`signal_has_edge = False`** — a credible weak/negative result,
> not a profitable edge.

The shipped default runs on a **synthetic** sentiment + price generator (no keys),
constructed so the in-sample edge decays out-of-sample and fails the DSR after
costs — the honest null by construction. The real-data path (Pushshift/PRAW +
Polygon point-in-time prices) lives behind the offline `ingest`+`score` CLI.

## Status

Scaffold: import-pure typed `src/` package with full contracts (signatures +
docstrings) for every module; the compute kernels are stubs raising
`NotImplementedError`, filled in by follow-up work.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'      # lean: NO transformers/torch/TF
# offline ingestion adapters: add the `ingest` extra (praw)
```

```python
import wsb_sentiment   # import-pure: no praw / network / torch at import
```

## Design

- **Lexicon sentiment only** — finance-augmented VADER (primary) with a TextBlob
  cross-check. No transformers/torch/TF; no model fit; no NLTK download at import.
- **Offline ingestion** — Pushshift/PRAW adapters are batch tools behind the
  `[ingest]` extra; they are never called at request time.
- **Leakage guards** — strict as-of cutoff at the prior session close,
  `signal.shift(1)`, forward-return labels only, `pct_change(fill_method=None)`,
  a **train-only** standardizer, walk-forward purge + embargo, and a
  **point-in-time** S&P 500 universe (no future-constituent selection).
- **Honest stats** — Deflated/Probabilistic Sharpe with an *effective* trial count
  (PCA of the swept grid), PBO via CSCV, HAC (Newey-West) t-stats, and the
  Memmel-JK test versus buy-and-hold. The `signal_has_edge` verdict is a **pure
  function** of OOS net Sharpe + DSR + PBO + HAC.

## Validation

| Check | What it pins |
| --- | --- |
| parity | VADER vs TextBlob agreement bands; daily roll-up vs a slow reference; DSR/PSR vs the reused `dsr.py` to 1e-10 |
| property | as-of prefix-determinism, shift-equivariance, standardization scale-invariance, mention-count monotonicity (Hypothesis) |
| regression | golden decaying-signal backtest (in-sample edge → OOS decay → `signal_has_edge = False` after DSR/costs); no-lookahead golden test |
| integration | end-to-end score → aggregate → signal → backtest on synthetic data |

## Limitations

- **Pushshift deletion bias + coverage gap** — removed/deleted posts are missing
  and coverage thins after 2023, biasing historical sentiment.
- **PIT survivorship** — meme tickers outside the as-of S&P 500 universe are
  **descriptive-only** and never traded.
- **Synthetic default** — the shipped tool runs on a synthetic generator so the
  result is reproducible and the null is honest; real data via the ingest path.

## References

- DeMiguel, Garlappi & Uppal (2009) — out-of-sample decay of estimated strategies.
- Bailey & López de Prado (2014) — the Deflated Sharpe Ratio.
- Bailey, Borwein, López de Prado & Zhu (2017) — Probability of Backtest
  Overfitting (CSCV).
- Hutto & Gilbert (2014) — VADER lexicon-based sentiment.

## License

MIT — see [LICENSE](LICENSE).
