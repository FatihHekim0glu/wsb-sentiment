"""Honest-statistics layer for the sentiment signal.

Bundles the overfitting / significance guards the verdict depends on:

- the DEFLATED / PROBABILISTIC Sharpe ratio with an EFFECTIVE number of trials
  estimated by PCA of the trial-return matrix over the swept
  lexicon x window x lag x threshold x cost grid (so correlated trials do not
  inflate the multiplicity count);
- PROBABILITY OF BACKTEST OVERFITTING via CSCV (:func:`pbo_cscv`);
- a HAC (Newey-West) t-statistic on the net OOS returns;
- the MEMMEL-JK test of the signal's Sharpe against the buy-and-hold baseline.

This module orchestrates the vendored primitives (``dsr``, ``pbo``, ``hac``,
``memmel``, ``bootstrap_ci``); it adds no new heavy math beyond the PCA-of-trials
effective-trials estimator. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from wsb_sentiment._typing import FloatArray


@dataclass(frozen=True, slots=True)
class HonestStats:
    """The full honest-statistics bundle the verdict is derived from.

    Attributes
    ----------
    net_sharpe:
        Annualized net (after-cost) OOS Sharpe of the selected signal.
    buyhold_sharpe:
        Annualized OOS Sharpe of the equal-weight buy-and-hold baseline.
    deflated_sharpe:
        The Deflated Sharpe Ratio in ``[0, 1]`` (FULL effective-trials grid).
    psr:
        The Probabilistic Sharpe Ratio in ``[0, 1]`` vs a zero benchmark.
    pbo:
        The Probability of Backtest Overfitting in ``[0, 1]`` (CSCV).
    hac_tstat:
        The Newey-West HAC t-statistic on the net OOS mean return.
    hac_pvalue:
        The two-sided p-value for ``hac_tstat``.
    n_effective_trials:
        The PCA-estimated effective number of independent trials.
    n_obs:
        The number of OOS observations.
    """

    net_sharpe: float
    buyhold_sharpe: float
    deflated_sharpe: float
    psr: float
    pbo: float
    hac_tstat: float
    hac_pvalue: float
    n_effective_trials: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the bundle."""
        return asdict(self)


def effective_n_trials(trial_returns: pd.DataFrame, *, var_threshold: float = 0.95) -> float:
    """Estimate the EFFECTIVE number of independent trials by PCA of trial returns.

    Many configurations on the swept grid (lexicon x window x lag x threshold x
    cost) produce highly-correlated return streams, so the raw grid size grossly
    overstates the multiplicity. We PCA the standardized trial-return matrix and
    return the number of principal components needed to explain ``var_threshold``
    of the variance — a deflation-honest effective trial count that feeds the DSR.

    Parameters
    ----------
    trial_returns:
        A ``(T, N_trials)`` matrix; each column is one configuration's OOS return
        series.
    var_threshold:
        The cumulative-variance fraction defining "enough" components
        (default ``0.95``).

    Returns
    -------
    float
        The effective number of trials (``>= 1``).

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("effective_n_trials is not yet implemented")


def hac_tstat(returns: pd.Series | FloatArray, *, lag: int | None = None) -> tuple[float, float]:
    """Newey-West HAC t-statistic and two-sided p-value for the mean return.

    Wraps the vendored :func:`wsb_sentiment.evaluation.hac.newey_west_se` to form
    ``t = mean / HAC_se(mean)`` and a normal-approximation two-sided p-value, so
    serial correlation in the daily OOS returns does not inflate significance.

    Parameters
    ----------
    returns:
        The net OOS return series.
    lag:
        Bartlett lag truncation; ``None`` selects the Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        The HAC t-statistic and its two-sided p-value.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("hac_tstat is not yet implemented")


def compute_honest_stats(
    net_returns: pd.Series,
    buyhold_returns: pd.Series,
    trial_returns: pd.DataFrame,
    *,
    n_grid_trials: int,
    periods_per_year: int = 252,
) -> HonestStats:
    """Assemble the full honest-statistics bundle from OOS returns and the trial grid.

    Computes net/buy-hold Sharpe, the PCA-effective trial count, the DSR/PSR
    (vendored ``dsr``), the PBO via CSCV (vendored ``pbo``), the HAC t-stat
    (:func:`hac_tstat`), and the Memmel-JK comparison vs buy-and-hold (vendored
    ``memmel``), returning a single immutable :class:`HonestStats`.

    Parameters
    ----------
    net_returns:
        The net (after-cost) OOS return series of the selected signal.
    buyhold_returns:
        The OOS return series of the equal-weight buy-and-hold baseline.
    trial_returns:
        The ``(T, N_grid)`` matrix of all swept configurations' OOS returns (for
        PBO and the effective-trials estimate).
    n_grid_trials:
        The raw number of configurations swept (the upper bound on multiplicity).
    periods_per_year:
        Annualization factor (``252`` for daily).

    Returns
    -------
    HonestStats
        The full bundle the verdict consumes.

    Raises
    ------
    NotImplementedError
        This is a typed stub awaiting implementation.
    """
    raise NotImplementedError("compute_honest_stats is not yet implemented")
