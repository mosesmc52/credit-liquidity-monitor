import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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


def classify_stress_regime(stress_prob: float) -> str:
    if stress_prob >= 0.85:
        return "CRISIS"
    if stress_prob >= 0.75:
        return "STRESS"
    if stress_prob >= 0.60:
        return "WATCH"
    return "NORMAL"


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
