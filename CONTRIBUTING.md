# Contributing

Thanks for your interest in `wsb-sentiment`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the lean extras + dev tooling.
uv venv
uv pip install -e '.[data,viz,dev]'
# (The offline ingestion adapters need the extra praw dependency: add `ingest`.)
```

The package is **import-pure**: `import wsb_sentiment` must NOT import praw,
vaderSentiment, textblob, plotly, or typer, and must trigger no network call. Heavy
dependencies are imported lazily inside the functions that need them; ingestion and
scoring are OFFLINE batch paths that never run at request time.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src                                                     # lint
uv run mypy src                                                           # types (strict)
uv run pytest -q --cov=wsb_sentiment --cov-report=term --cov-fail-under=85  # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) must pass.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Honest-null discipline

The headline is a credible WEAK/NEGATIVE result: the naive VADER WSB sentiment
signal's mild in-sample correlation with next-day returns largely decays
out-of-sample and fails the Deflated Sharpe and per-side cost hurdles. The
`signal_has_edge` verdict is a PURE function of the OOS net Sharpe, DSR, PBO, and
HAC significance — never narrated. Do not weaken the leakage guards (as-of cutoff,
`signal.shift`, train-only scaler, PIT universe) or overclaim profit.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers — no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
