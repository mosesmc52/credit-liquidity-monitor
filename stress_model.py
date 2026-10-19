"""Transparent, uncalibrated credit and funding stress indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from helpers import fresh_reindex, logistic_transform, rolling_zscore


WEIGHTS = {
    "hy": 0.30,
    "bbb": 0.20,
    "dispersion": 0.05,
    "funding": 0.35,
    "rrp": 0.05,
    "repo": 0.05,
}


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Require a usable baseline; a flat series has a neutral relative move."""
    minimum = min(20, window)
    z = rolling_zscore(s, window)
    count = s.notna().rolling(window, min_periods=1).sum()
    std = s.rolling(window, min_periods=minimum).std(ddof=0)
    z = z.where(std.ne(0), 0.0).where(count >= minimum)
    return z.replace([np.inf, -np.inf], np.nan)


def _level_floor(value: pd.Series, threshold: float) -> pd.Series:
    """Keep an elevated spread visible after the rolling baseline catches up."""
    if threshold <= 0:
        raise ValueError("level threshold must be positive")
    lower = 0.75 * threshold
    return ((value - lower) / (threshold - lower) * 1.75).clip(lower=0, upper=3.5)


def _level_adjusted_z(z: pd.Series, value: pd.Series, threshold: float) -> pd.Series:
    floor = _level_floor(value, threshold)
    elevated = pd.concat([z, floor], axis=1).max(axis=1, skipna=False)
    return z.where(value <= 0.75 * threshold, elevated)


def compute_stress_indicator(
    hy_oas: pd.Series,
    bbb_oas: pd.Series,
    sofr: pd.Series,
    effr: pd.Series,
    rrp_total: pd.Series,
    repo_total: pd.Series | None = None,
    *,
    rolling_window: int = 60,
    hy_threshold: float = 6.0,
    bbb_threshold: float = 2.5,
    funding_threshold_bps: float = 10.0,
    max_business_days: int = 3,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return score, 0–100 display index and coverage on observed market dates.

    HY, BBB, SOFR and EFFR are required. RRP and repo are optional and their
    weights are removed when unavailable. Every input is used only after its
    observation date and for at most ``max_business_days`` business days.
    """
    if rolling_window < 5:
        raise ValueError("rolling_window must be at least 5")
    if max_business_days < 0:
        raise ValueError("max_business_days must not be negative")

    index = hy_oas.index.union(bbb_oas.index).union(sofr.index).union(effr.index)
    if as_of is not None:
        index = index.union(pd.DatetimeIndex([pd.Timestamp(as_of)]))
    index = pd.DatetimeIndex(index).sort_values()
    if index.empty:
        return pd.DataFrame(index=index, columns=["stress_score", "stress_index", "coverage"], dtype=float)

    hy = fresh_reindex(hy_oas, index, max_business_days)
    bbb = fresh_reindex(bbb_oas, index, max_business_days)
    sofr_aligned = fresh_reindex(sofr, index, max_business_days)
    effr_aligned = fresh_reindex(effr, index, max_business_days)
    dispersion = hy - bbb
    funding = (sofr_aligned - effr_aligned) * 100.0

    components = {
        "hy": _level_adjusted_z(
            fresh_reindex(_zscore(hy_oas, rolling_window), index, max_business_days),
            hy, hy_threshold,
        ),
        "bbb": _level_adjusted_z(
            fresh_reindex(_zscore(bbb_oas, rolling_window), index, max_business_days),
            bbb, bbb_threshold,
        ),
        "dispersion": _zscore(dispersion, rolling_window),
        "funding": _level_adjusted_z(
            _zscore(funding, rolling_window), funding, funding_threshold_bps,
        ),
        "rrp": fresh_reindex(_zscore(rrp_total, rolling_window), index, max_business_days),
    }
    if repo_total is not None and not repo_total.empty:
        components["repo"] = fresh_reindex(_zscore(repo_total, rolling_window), index, max_business_days)

    frame = pd.DataFrame(components, index=index)
    required = frame[["hy", "bbb", "dispersion", "funding"]].notna().all(axis=1)
    weighted = frame.mul(pd.Series(WEIGHTS), axis=1)
    available_weight = frame.notna().mul(pd.Series(WEIGHTS), axis=1).sum(axis=1)
    numerator = pd.to_numeric(weighted.sum(axis=1, min_count=1), errors="coerce")
    score = numerator.div(available_weight.replace(0, np.nan)).where(required)
    result = pd.DataFrame(index=index)
    result["stress_score"] = score
    result["stress_index"] = logistic_transform(score) * 100.0
    configured_weight = sum(WEIGHTS[name] for name in frame.columns)
    result["coverage"] = available_weight.div(configured_weight).where(required, 0.0)
    return result
