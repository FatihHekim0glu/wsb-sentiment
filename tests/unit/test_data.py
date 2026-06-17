"""Unit tests for the synthetic sentiment + price generator and loaders.

Covers the determinism, value-domain, decay-structure, no-lookahead returns,
loader resolution, and point-in-time universe guarantees of
:mod:`wsb_sentiment.data`.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment.data import (
    SyntheticPanel,
    compute_returns,
    generate_synthetic_panel,
    load_sentiment_panel,
    pit_universe,
)

_TICKERS = ["GME", "AMC", "TSLA", "AAPL", "NVDA"]
_START = date(2021, 1, 4)
_END = date(2022, 12, 30)


def _lag_corr(sentiment: pd.DataFrame, returns: pd.DataFrame, sl: slice) -> float:
    """Pooled correlation between PRIOR-day sentiment and the current return."""
    s = sentiment.iloc[sl].shift(1).to_numpy().ravel()
    r = returns.iloc[sl].to_numpy().ravel()
    mask = np.isfinite(s) & np.isfinite(r)
    return float(np.corrcoef(s[mask], r[mask])[0, 1])


@pytest.mark.unit
def test_generate_is_deterministic() -> None:
    """The same ``(tickers, start, end, seed)`` reproduces a byte-identical panel."""
    a = generate_synthetic_panel(_TICKERS, _START, _END, seed=7)
    b = generate_synthetic_panel(_TICKERS, _START, _END, seed=7)
    pd.testing.assert_frame_equal(a.mean_compound, b.mean_compound)
    pd.testing.assert_frame_equal(a.mention_count, b.mention_count)
    pd.testing.assert_frame_equal(a.positive_share, b.positive_share)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    pd.testing.assert_frame_equal(a.universe_mask, b.universe_mask)


@pytest.mark.unit
def test_generate_is_seed_sensitive() -> None:
    """Different seeds produce different (but still reproducible) panels."""
    a = generate_synthetic_panel(_TICKERS, _START, _END, seed=7)
    b = generate_synthetic_panel(_TICKERS, _START, _END, seed=8)
    assert not a.prices.equals(b.prices)
    assert not a.mean_compound.equals(b.mean_compound)


@pytest.mark.unit
def test_generated_values_are_in_domain() -> None:
    """Compound in [-1, 1], positive-share in [0, 1], mentions non-negative ints,
    prices strictly positive."""
    panel = generate_synthetic_panel(_TICKERS, _START, _END, seed=11)
    mc = panel.mean_compound.to_numpy()
    ps = panel.positive_share.to_numpy()
    mn = panel.mention_count.to_numpy()
    assert mc.min() >= -1.0 and mc.max() <= 1.0
    assert ps.min() >= 0.0 and ps.max() <= 1.0
    assert mn.min() >= 0.0
    assert np.allclose(mn, np.round(mn))  # integer-valued counts
    assert panel.prices.to_numpy().min() > 0.0
    assert panel.data_source == "synthetic"


@pytest.mark.unit
def test_panel_shapes_align() -> None:
    """All emitted panels share the same index and columns."""
    panel = generate_synthetic_panel(_TICKERS, _START, _END, seed=3)
    index = panel.mean_compound.index
    cols = list(panel.mean_compound.columns)
    for frame in (
        panel.mention_count,
        panel.positive_share,
        panel.prices,
        panel.universe_mask,
    ):
        assert frame.index.equals(index)
        assert list(frame.columns) == cols
    assert cols == _TICKERS


@pytest.mark.unit
def test_in_sample_edge_decays_out_of_sample() -> None:
    """The lagged sentiment->return correlation is clearly larger in-sample.

    This is the honest-null structure BY CONSTRUCTION: a mild in-sample
    predictability that largely vanishes out-of-sample.
    """
    panel = generate_synthetic_panel(_TICKERS, _START, _END, seed=7)
    returns = panel.prices.pct_change(fill_method=None)
    n = len(panel.mean_compound)
    half = n // 2
    in_sample = _lag_corr(panel.mean_compound, returns, slice(0, half))
    out_of_sample = _lag_corr(panel.mean_compound, returns, slice(half, n))
    assert in_sample > 0.0
    assert abs(out_of_sample) < abs(in_sample)


@pytest.mark.unit
def test_to_dict_is_json_serializable() -> None:
    """``SyntheticPanel.to_dict`` round-trips through ``json.dumps`` (NaN -> None)."""
    panel = generate_synthetic_panel(["GME", "AMC"], _START, date(2021, 3, 31), seed=2)
    payload = panel.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored["data_source"] == "synthetic"
    assert set(restored["mean_compound"]) == set(payload["mean_compound"])
    # universe_mask values are booleans, not floats.
    first_day = next(iter(restored["universe_mask"].values()))
    assert all(isinstance(v, bool) for v in first_day.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"in_sample_corr": float("nan")}, "finite"),
        ({"mention_lambda": -1.0}, "positive"),
        ({"decay_at": 0.0}, "decay_at"),
        ({"decay_at": 1.5}, "decay_at"),
    ],
)
def test_generate_rejects_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    """Bad coupling / lambda / decay arguments raise :class:`ValidationError`."""
    with pytest.raises(ValidationError, match=match):
        generate_synthetic_panel(_TICKERS, _START, _END, **kwargs)


@pytest.mark.unit
def test_generate_rejects_empty_and_duplicate_tickers() -> None:
    """An empty ticker set, an empty date range, or duplicate tickers raise."""
    with pytest.raises(ValidationError, match="non-empty"):
        generate_synthetic_panel([], _START, _END)
    with pytest.raises(ValidationError, match="unique"):
        generate_synthetic_panel(["GME", "GME"], _START, _END)
    with pytest.raises(ValidationError, match="empty date range"):
        generate_synthetic_panel(_TICKERS, date(2021, 1, 9), date(2021, 1, 10))  # Sat-Sun


@pytest.mark.unit
def test_compute_returns_is_forward_safe() -> None:
    """Returns use ``pct_change(fill_method=None)`` and drop only the leading row."""
    panel = generate_synthetic_panel(_TICKERS, _START, date(2021, 6, 30), seed=5)
    returns = compute_returns(panel.prices)
    assert len(returns) == len(panel.prices) - 1
    # First retained return matches a manual no-ffill pct_change.
    expected = panel.prices.pct_change(fill_method=None).iloc[1]
    pd.testing.assert_series_equal(returns.iloc[0], expected)


@pytest.mark.unit
def test_compute_returns_does_not_forward_fill_across_gaps() -> None:
    """A NaN price gap stays a NaN return — it is NOT manufactured into a 0."""
    prices = pd.DataFrame(
        {"X": [100.0, np.nan, 110.0], "Y": [50.0, 55.0, 60.5]},
        index=pd.date_range("2022-01-03", periods=3, freq="B"),
    )
    returns = compute_returns(prices)
    # With fill_method=None, the gap row's X return is NaN (no spurious zero).
    assert pd.isna(returns["X"].iloc[0])
    # Y is gap-free and computes normally.
    assert returns["Y"].iloc[0] == pytest.approx(0.1)


@pytest.mark.unit
def test_load_sentiment_panel_synthetic_default() -> None:
    """The default loader resolves to the deterministic synthetic panel."""
    panel, source = load_sentiment_panel(["GME", "AMC"], _START, date(2021, 6, 30))
    assert source == "synthetic"
    assert isinstance(panel, SyntheticPanel)
    assert panel.data_source == "synthetic"
    # Reproducible: a second load with the same seed is identical.
    again, _ = load_sentiment_panel(["GME", "AMC"], _START, date(2021, 6, 30))
    pd.testing.assert_frame_equal(panel.prices, again.prices)


@pytest.mark.unit
def test_load_sentiment_panel_cache_miss_falls_back(tmp_path: object) -> None:
    """An explicit cache request with no parquet falls back to synthetic."""
    panel, source = load_sentiment_panel(
        ["GME"], _START, date(2021, 3, 31), source_pref="cache", cache_path=str(tmp_path)
    )
    assert source == "synthetic"
    assert panel.data_source == "synthetic"


@pytest.mark.unit
def test_load_sentiment_panel_reads_cache(tmp_path: object) -> None:
    """A precomputed parquet bundle is read back as a ``cache`` panel."""
    import os

    base = generate_synthetic_panel(["GME", "AMC"], _START, date(2021, 6, 30), seed=4)
    cache_dir = str(tmp_path)
    base.mean_compound.to_parquet(os.path.join(cache_dir, "mean_compound.parquet"))
    base.mention_count.to_parquet(os.path.join(cache_dir, "mention_count.parquet"))
    base.positive_share.to_parquet(os.path.join(cache_dir, "positive_share.parquet"))
    base.prices.to_parquet(os.path.join(cache_dir, "prices.parquet"))

    panel, source = load_sentiment_panel(
        ["GME", "AMC"], _START, date(2021, 6, 30), source_pref="cache", cache_path=cache_dir
    )
    assert source == "cache"
    assert panel.data_source == "cache"
    # Parquet does not preserve the DatetimeIndex ``freq`` attribute; compare
    # values (and the index labels) without the frequency metadata.
    pd.testing.assert_frame_equal(panel.prices, base.prices, check_freq=False)


@pytest.mark.unit
def test_load_sentiment_panel_cache_with_no_matching_tickers_falls_back(tmp_path: object) -> None:
    """A cache whose columns miss every requested ticker falls back to synthetic."""
    import os

    base = generate_synthetic_panel(["AAA", "BBB"], _START, date(2021, 3, 31), seed=1)
    cache_dir = str(tmp_path)
    for name, frame in (
        ("mean_compound", base.mean_compound),
        ("mention_count", base.mention_count),
        ("positive_share", base.positive_share),
        ("prices", base.prices),
    ):
        frame.to_parquet(os.path.join(cache_dir, f"{name}.parquet"))

    # Request tickers absent from the cache -> fall back to synthetic.
    panel, source = load_sentiment_panel(
        ["GME"], _START, date(2021, 3, 31), source_pref="auto", cache_path=cache_dir
    )
    assert source == "synthetic"
    assert panel.data_source == "synthetic"


@pytest.mark.unit
def test_load_sentiment_panel_cache_outside_window_falls_back(tmp_path: object) -> None:
    """A cache with no rows in the requested window falls back to synthetic."""
    import os

    base = generate_synthetic_panel(["GME"], date(2020, 1, 2), date(2020, 3, 31), seed=1)
    cache_dir = str(tmp_path)
    for name, frame in (
        ("mean_compound", base.mean_compound),
        ("mention_count", base.mention_count),
        ("positive_share", base.positive_share),
        ("prices", base.prices),
    ):
        frame.to_parquet(os.path.join(cache_dir, f"{name}.parquet"))

    # Request a 2021 window that the 2020 cache does not cover.
    _panel, source = load_sentiment_panel(
        ["GME"], _START, date(2021, 3, 31), source_pref="cache", cache_path=cache_dir
    )
    assert source == "synthetic"


@pytest.mark.unit
def test_load_sentiment_panel_corrupt_cache_falls_back(tmp_path: object) -> None:
    """A corrupt/unreadable cache file is swallowed and falls back to synthetic."""
    import os

    cache_dir = str(tmp_path)
    for name in ("mean_compound", "mention_count", "positive_share", "prices"):
        with open(os.path.join(cache_dir, f"{name}.parquet"), "w", encoding="utf-8") as handle:
            handle.write("not a parquet file")

    panel, source = load_sentiment_panel(
        ["GME"], _START, date(2021, 3, 31), source_pref="cache", cache_path=cache_dir
    )
    assert source == "synthetic"
    assert panel.data_source == "synthetic"


@pytest.mark.unit
def test_load_sentiment_panel_rejects_bad_source() -> None:
    """An unsupported ``source_pref`` raises :class:`ValidationError`."""
    with pytest.raises(ValidationError, match="source_pref"):
        load_sentiment_panel(["GME"], _START, _END, source_pref="polygon")  # type: ignore[arg-type]


@pytest.mark.unit
def test_pit_universe_is_monotone_and_pit_safe() -> None:
    """Membership is monotone non-decreasing per ticker (no future-driven removal)."""
    sessions = pd.date_range(_START, _END, freq="B")
    mask = pit_universe(_TICKERS, sessions)
    assert mask.shape == (len(sessions), len(_TICKERS))
    assert all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes)
    for ticker in _TICKERS:
        col = mask[ticker].to_numpy().astype(int)
        # Once admitted, a ticker stays admitted (diffs are never negative).
        assert (np.diff(col) >= 0).all()


@pytest.mark.unit
def test_pit_universe_is_deterministic() -> None:
    """The PIT mask depends only on the symbols and the session calendar."""
    sessions = pd.date_range(_START, date(2021, 12, 31), freq="B")
    a = pit_universe(_TICKERS, sessions)
    b = pit_universe(_TICKERS, sessions)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_pit_universe_rejects_bad_input() -> None:
    """Empty tickers / sessions / unsupported source raise :class:`ValidationError`."""
    sessions = pd.date_range(_START, date(2021, 3, 31), freq="B")
    with pytest.raises(ValidationError, match="non-empty"):
        pit_universe([], sessions)
    with pytest.raises(ValidationError, match="non-empty"):
        pit_universe(_TICKERS, pd.DatetimeIndex([]))
    with pytest.raises(ValidationError, match="source_pref"):
        pit_universe(_TICKERS, sessions, source_pref="polygon")  # type: ignore[arg-type]
