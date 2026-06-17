# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/DESIGN.md` — full architecture, post→verdict data flow, leakage invariants,
  and testing strategy.
- Architecture Decision Records under `docs/decisions/`: as-of cutoff at the prior
  session close (0001), point-in-time universe with no future-constituent selection
  (0002), train-only standardizer (0003), the mechanically-enforced honest weak null
  (0004), and the synthetic default with no live ingestion at request time (0005).
- `CITATION.cff` (Hutto–Gilbert VADER, Bailey–López de Prado DSR/PBO, DeMiguel et al.
  OOS-decay, Newey–West HAC).

### Changed

- README: honest weak/negative headline with the actual synthetic metrics
  (OOS net Sharpe 0.29, DSR 0.26, PBO 0.39, HAC p 0.79 → `signal_has_edge = False`),
  an expanded Validation table (DSR/PSR 1e-10 parity; VADER–TextBlob sign-agreement
  bands; oracle→test wording), a Reproduce block, ADR cross-links, and refreshed
  status (compute core implemented; 290 tests, ≈93% coverage).

## [0.1.0] - 2026-06-17

### Added

- Initial package skeleton (src-layout, import name `wsb_sentiment`, `py.typed`).
- Reused infra from `hrp-portfolio`: `_constants`, `_typing`, `_exceptions`
  (`WsbSentimentError` base), `_validation`, `_manifest` (`RunManifest` with
  BLAKE2b config-hash), `_rng` (seeded PCG64 generator + substream spawning);
  the no-lookahead `backtest.walk_forward` + `backtest.costs` + `backtest.stats`;
  and `evaluation.dsr` (Deflated / Probabilistic Sharpe).
- Vendored honest-stats primitives (PBO/CSCV, HAC/Newey-West, Memmel-JK,
  stationary-bootstrap CI) from `pairs_trading.evaluation`, re-homed onto the
  package's own exception base.
- Stub signatures with full contracts for the new subpackages: `ingest`
  (`pushshift`, `reddit_api`, `extract`), `nlp` (`vader`, `textblob_parity`),
  `aggregate.rollup`, `signal.build`, `backtest.engine`, and `evaluation`
  (`stats`, `verdict`).
- Synthetic sentiment + price generator and loaders in `data.py` (honest null by
  construction: the in-sample edge decays out-of-sample and fails the Deflated
  Sharpe after costs), a PIT-universe hook, lazy Plotly figure builders, and a
  Typer CLI stub.
- Seeded test fixtures (`synthetic_sentiment_panel`, `decaying_signal`,
  `pure_noise`) and a partitioned `tests/` tree
  (unit / parity / property / regression / integration).

[Unreleased]: https://github.com/FatihHekim0glu/wsb-sentiment/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/wsb-sentiment/releases/tag/v0.1.0
