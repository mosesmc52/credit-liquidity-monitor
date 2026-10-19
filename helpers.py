import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

STRESS_INDEX_WATCH = 60.0
STRESS_INDEX_STRESS = 75.0
STRESS_INDEX_CRISIS = 85.0


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def getenv_int(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def getenv_float(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def last_n_above(s: pd.Series, threshold: float, n: int) -> bool:
    s = s.dropna()
    if len(s) < n:
        return False
    return bool((s.iloc[-n:] >= threshold).all())


def zero_series_like(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype=float)


def logistic_transform(s: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-s))


def classify_stress_regime(stress_index: float) -> str:
    if pd.isna(stress_index):
        return "UNKNOWN"
    if stress_index >= STRESS_INDEX_CRISIS:
        return "CRISIS"
    if stress_index >= STRESS_INDEX_STRESS:
        return "STRESS"
    if stress_index >= STRESS_INDEX_WATCH:
        return "WATCH"
    return "NORMAL"


def fresh_reindex(s: pd.Series, index: pd.DatetimeIndex, max_business_days: int = 3) -> pd.Series:
    """Carry an observation forward only while its source date is recent."""
    clean = s.dropna().sort_index()
    if clean.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    combined = index.union(clean.index)
    values = clean.reindex(combined).ffill().reindex(index)
    observed_at = pd.Series(clean.index, index=clean.index).reindex(combined).ffill().reindex(index)
    earliest = pd.Series(index - pd.offsets.BDay(max_business_days), index=index)
    return values.where(observed_at >= earliest)


def last_n_days(s: pd.Series, days: int = 90) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s

    cutoff = s.index.max() - pd.Timedelta(days=days)
    return s[s.index >= cutoff]


# =========================
# Analytics
# =========================


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    clean = s.dropna()
    if clean.empty:
        return s.astype(float)

    effective_window = min(window, len(clean))
    min_periods = min(effective_window, 5)
    mu = clean.rolling(effective_window, min_periods=min_periods).mean()
    sigma = clean.rolling(effective_window, min_periods=min_periods).std(ddof=0)
    z = (clean - mu) / sigma.replace(0.0, pd.NA)
    return z.reindex(s.index)


def latest_valid(s: pd.Series) -> Optional[float]:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def latest_date(s: pd.Series) -> Optional[pd.Timestamp]:
    s = s.dropna()
    if s.empty:
        return None
    return s.index[-1]
