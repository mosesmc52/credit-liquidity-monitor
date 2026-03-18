import os


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


def build_latent_stress_signal(signals: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Fast latent stress proxy:
    combines standardized credit + liquidity signals into a single score.
    This is a production-friendly stand-in for a fuller PyMC latent model.
    """
    cols = []

    if "hy_oas_z" in signals:
        cols.append(signals["hy_oas_z"].rename("hy_oas_z"))
    if "bbb_oas_z" in signals:
        cols.append(signals["bbb_oas_z"].rename("bbb_oas_z"))
    if "hy_minus_bbb_z" in signals:
        cols.append(signals["hy_minus_bbb_z"].rename("hy_minus_bbb_z"))

    # liquidity
    spread_z = rolling_zscore(signals["sofr_effr_spread_bps"], ROLLING_WINDOW)
    cols.append(spread_z.rename("sofr_effr_z"))

    if "rrp_total_z" in signals:
        cols.append(signals["rrp_total_z"].rename("rrp_total_z"))
    if "repo_total_z" in signals:
        cols.append(signals["repo_total_z"].rename("repo_total_z"))

    X = pd.concat(cols, axis=1).sort_index()

    # Robust cleanup
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    X = X.ffill()

    # Default weights: heavier on credit, lighter on plumbing/context
    weights = {
        "hy_oas_z": 0.30,
        "bbb_oas_z": 0.15,
        "hy_minus_bbb_z": 0.20,
        "sofr_effr_z": 0.20,
        "rrp_total_z": 0.10,
        "repo_total_z": 0.05,
    }

    usable = [c for c in X.columns if c in weights]
    w = np.array([weights[c] for c in usable], dtype=float)
    w = w / w.sum()

    raw_score = X[usable].mul(w, axis=1).sum(axis=1)
    smoothed_score = raw_score.ewm(span=5, adjust=False).mean()

    # Convert latent score to 0-1 probability
    # score ~ 0 => 50%, score 1.5+ => high stress probability
    stress_prob = pd.Series(
        expit(smoothed_score), index=smoothed_score.index, name="stress_prob"
    )

    out = pd.DataFrame(
        {
            "latent_stress_raw": raw_score,
            "latent_stress": smoothed_score,
            "stress_prob": stress_prob,
        }
    )
    return out
