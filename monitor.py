from __future__ import annotations

import datetime as dt
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fredapi import Fred
from helpers import (
    classify_stress_regime,
    getenv_float,
    getenv_int,
    last_n_above,
    last_n_days,
    latest_date,
    latest_valid,
    logistic_transform,
    rolling_zscore,
    str2bool,
    zero_series_like,
)
from nyfed_client import FetchSpec, NYFedClient
from SES import AmazonSES

load_dotenv(find_dotenv())

fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))

FRED_API_KEY = os.environ["FRED_API_KEY"]

load_dotenv(find_dotenv())


@dataclass
class AlertEvent:
    category: str
    name: str
    severity: str
    message: str


# =========================
# Configuration
# =========================


NYFED_BASE_URL = os.getenv("NYFED_BASE_URL", "https://markets.newyorkfed.org/api")

SES_REGION = os.environ["AWS_SES_REGION_NAME"]
SES_ACCESS_KEY = os.environ["AWS_SES_ACCESS_KEY_ID"]
SES_SECRET_KEY = os.environ["AWS_SES_SECRET_ACCESS_KEY"]
SES_FROM_ADDRESS = os.environ["FROM_ADDRESS"]
ALERT_TO = [x.strip() for x in os.environ["TO_ADDRESSES"].split(",") if x.strip()]

LOOKBACK_DAYS = 365
ROLLING_WINDOW = 60

HY_OAS_ABS_THRESHOLD = 6.0
BBB_OAS_ABS_THRESHOLD = 2.5
OAS_Z_THRESHOLD = 1.75

SOFR_EFFR_SPREAD_BPS_THRESHOLD = 10.0
RRP_Z_THRESHOLD = 2.25
REPO_Z_THRESHOLD = 2.25

STRESS_PROB_WATCH = 0.60
STRESS_PROB_STRESS = 0.75
STRESS_PROB_CRISIS = 0.85

LOOKBACK_DAYS = getenv_int(os.getenv("LOOKBACK_DAYS"), 365)
ROLLING_WINDOW = getenv_int(os.getenv("ROLLING_WINDOW"), 60)


# Credit thresholds
HY_OAS_ABS_THRESHOLD = getenv_float(os.getenv("HY_OAS_ABS_THRESHOLD"), 6.0)  # percent
BBB_OAS_ABS_THRESHOLD = getenv_float(os.getenv("BBB_OAS_ABS_THRESHOLD"), 2.5)  # percent
OAS_Z_THRESHOLD = getenv_float(os.getenv("OAS_Z_THRESHOLD"), 1.75)

# Liquidity thresholds
SOFR_EFFR_SPREAD_BPS_THRESHOLD = getenv_float(
    os.getenv("SOFR_EFFR_SPREAD_BPS_THRESHOLD"), 10.0
)
RRP_Z_THRESHOLD = getenv_float(os.getenv("RRP_Z_THRESHOLD"), 2.25)
REPO_Z_THRESHOLD = getenv_float(os.getenv("REPO_Z_THRESHOLD"), 2.25)

SEND_ONLY_ON_ALERT = str2bool(os.getenv("SEND_ONLY_ON_ALERT", True))
INCLUDE_REPO_TOTAL = str2bool(os.getenv("INCLUDE_REPO_TOTAL", False))

CHART_LOOKBACK_DAYS = getenv_int(os.getenv("CHART_LOOKBACK_DAYS"), 90)


# =========================
# FRED helper
# =========================
def fetch_fred_series(series_id: str, start_date, end_date) -> pd.Series:
    """
    Fetch a FRED series using fredapi and return a clean pandas Series.
    """
    s = fred.get_series(
        series_id, observation_start=start_date, observation_end=end_date
    )

    if s is None or s.empty:
        return pd.Series(dtype=float, name=series_id)

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = series_id
    return s


# =========================
# NY Fed helpers
# =========================
def fetch_nyfed_series(
    client: NYFedClient,
    dataset: str,
    key: str,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.Series:
    rows = client.fetch_series(
        FetchSpec(dataset=dataset, key=key),
        start_date=start_date,
        end_date=end_date,
    )
    if not rows:
        return pd.Series(dtype=float, name=key)

    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["value"].astype(float).sort_index()
    s.name = key
    return s


def analyze_credit_liquidity(
    hy_oas: pd.Series,
    bbb_oas: pd.Series,
    sofr: pd.Series,
    effr: pd.Series,
    rrp_total: pd.Series,
    repo_total: Optional[pd.Series] = None,
    rolling_window: int = 60,
) -> tuple[List[AlertEvent], Dict[str, pd.Series]]:
    signals: Dict[str, pd.Series] = {}
    alerts: List[AlertEvent] = []

    # ----------------------------
    # Credit signals
    # ----------------------------
    signals["hy_oas"] = hy_oas.sort_index()
    signals["bbb_oas"] = bbb_oas.sort_index()

    signals["hy_oas_z"] = rolling_zscore(signals["hy_oas"], rolling_window)
    signals["bbb_oas_z"] = rolling_zscore(signals["bbb_oas"], rolling_window)

    signals["hy_minus_bbb"] = signals["hy_oas"] - signals["bbb_oas"]
    signals["hy_minus_bbb_z"] = rolling_zscore(signals["hy_minus_bbb"], rolling_window)

    # ----------------------------
    # Liquidity signals
    # ----------------------------
    rates = pd.concat(
        [sofr.rename("SOFR"), effr.rename("EFFR")],
        axis=1,
        sort=False,
    ).sort_index()

    rates = rates.ffill()
    signals["sofr"] = rates["SOFR"]
    signals["effr"] = rates["EFFR"]

    signals["sofr_effr_spread_bps"] = (signals["sofr"] - signals["effr"]) * 100.0
    signals["sofr_effr_spread_bps_z"] = rolling_zscore(
        signals["sofr_effr_spread_bps"], rolling_window
    )

    signals["rrp_total"] = rrp_total.sort_index()
    signals["rrp_total_z"] = rolling_zscore(signals["rrp_total"], rolling_window)

    if repo_total is not None and not repo_total.empty:
        signals["repo_total"] = repo_total.sort_index()
        signals["repo_total_z"] = rolling_zscore(signals["repo_total"], rolling_window)

    # ----------------------------
    # Composite stress score
    # ----------------------------
    base_index = signals["hy_oas_z"].index

    repo_z = (
        signals["repo_total_z"]
        if "repo_total_z" in signals
        else zero_series_like(base_index)
    )
    repo_z = repo_z.reindex(base_index).fillna(0.0)

    stress_components = pd.concat(
        [
            signals["hy_oas_z"].rename("hy_oas_z"),
            signals["bbb_oas_z"].rename("bbb_oas_z"),
            signals["hy_minus_bbb_z"].rename("hy_minus_bbb_z"),
            signals["sofr_effr_spread_bps_z"].rename("sofr_effr_spread_bps_z"),
            signals["rrp_total_z"].reindex(base_index).rename("rrp_total_z"),
            repo_z.rename("repo_total_z"),
        ],
        axis=1,
    ).fillna(0.0)

    signals["stress_score"] = (
        0.35 * stress_components["hy_oas_z"]
        + 0.20 * stress_components["bbb_oas_z"]
        + 0.20 * stress_components["hy_minus_bbb_z"]
        + 0.15 * stress_components["sofr_effr_spread_bps_z"]
        + 0.05 * stress_components["rrp_total_z"]
        + 0.05 * stress_components["repo_total_z"]
    )

    signals["stress_prob"] = logistic_transform(signals["stress_score"])

    stress_prob_latest = latest_valid(signals["stress_prob"]) or 0.0
    regime = classify_stress_regime(stress_prob_latest)

    signals["stress_regime"] = pd.Series(
        [regime] * len(signals["stress_prob"]),
        index=signals["stress_prob"].index,
        dtype="object",
    )

    # ----------------------------
    # Credit alerts
    # ----------------------------
    hy_latest = latest_valid(signals["hy_oas"])
    hy_z_latest = latest_valid(signals["hy_oas_z"])

    if hy_latest is not None and hy_latest >= HY_OAS_ABS_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="credit",
                name="High Yield OAS",
                severity=(
                    "high" if hy_latest >= HY_OAS_ABS_THRESHOLD + 1.5 else "medium"
                ),
                message=f"HY OAS is {hy_latest:.2f}% (threshold {HY_OAS_ABS_THRESHOLD:.2f}%).",
            )
        )

    if hy_z_latest is not None and (
        hy_z_latest >= OAS_Z_THRESHOLD + 0.75
        or last_n_above(signals["hy_oas_z"], OAS_Z_THRESHOLD, 3)
    ):
        alerts.append(
            AlertEvent(
                category="credit",
                name="High Yield OAS z-score",
                severity="high" if hy_z_latest >= OAS_Z_THRESHOLD + 0.75 else "medium",
                message=f"HY OAS z-score is {hy_z_latest:.2f} (threshold {OAS_Z_THRESHOLD:.2f}).",
            )
        )

    bbb_latest = latest_valid(signals["bbb_oas"])
    bbb_z_latest = latest_valid(signals["bbb_oas_z"])

    if bbb_latest is not None and bbb_latest >= BBB_OAS_ABS_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="credit",
                name="BBB OAS",
                severity="medium",
                message=f"BBB OAS is {bbb_latest:.2f}% (threshold {BBB_OAS_ABS_THRESHOLD:.2f}%).",
            )
        )

    if bbb_z_latest is not None and (
        bbb_z_latest >= OAS_Z_THRESHOLD + 0.75
        or last_n_above(signals["bbb_oas_z"], OAS_Z_THRESHOLD, 3)
    ):
        alerts.append(
            AlertEvent(
                category="credit",
                name="BBB OAS z-score",
                severity="medium",
                message=f"BBB OAS z-score is {bbb_z_latest:.2f} (threshold {OAS_Z_THRESHOLD:.2f}).",
            )
        )

    disp_latest = latest_valid(signals["hy_minus_bbb"])
    disp_z_latest = latest_valid(signals["hy_minus_bbb_z"])

    if disp_z_latest is not None and (
        disp_z_latest >= OAS_Z_THRESHOLD + 0.50
        or last_n_above(signals["hy_minus_bbb_z"], OAS_Z_THRESHOLD, 3)
    ):
        alerts.append(
            AlertEvent(
                category="credit",
                name="HY vs BBB dispersion",
                severity="high" if disp_z_latest >= OAS_Z_THRESHOLD + 1.0 else "medium",
                message=(
                    f"HY-BBB spread is {disp_latest:.2f}% and z-score is {disp_z_latest:.2f} "
                    f"(threshold {OAS_Z_THRESHOLD:.2f})."
                ),
            )
        )

    # ----------------------------
    # Liquidity alerts
    # ----------------------------
    spread_latest = latest_valid(signals["sofr_effr_spread_bps"])
    spread_z_latest = latest_valid(signals["sofr_effr_spread_bps_z"])

    if spread_latest is not None and spread_latest >= SOFR_EFFR_SPREAD_BPS_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="liquidity",
                name="SOFR-EFFR spread",
                severity="high" if spread_latest >= 20.0 else "medium",
                message=(
                    f"SOFR-EFFR spread is {spread_latest:.1f} bps "
                    f"(threshold {SOFR_EFFR_SPREAD_BPS_THRESHOLD:.1f} bps)."
                ),
            )
        )

    if spread_z_latest is not None and (
        spread_z_latest >= 2.5
        or last_n_above(signals["sofr_effr_spread_bps_z"], 2.0, 2)
    ):
        alerts.append(
            AlertEvent(
                category="liquidity",
                name="SOFR-EFFR z-score",
                severity="high" if spread_z_latest >= 3.0 else "medium",
                message=f"SOFR-EFFR z-score is {spread_z_latest:.2f} (threshold 2.00).",
            )
        )

    rrp_z_latest = latest_valid(signals["rrp_total_z"])
    if rrp_z_latest is not None and (
        rrp_z_latest >= RRP_Z_THRESHOLD + 0.50
        or last_n_above(signals["rrp_total_z"], RRP_Z_THRESHOLD, 3)
    ):
        alerts.append(
            AlertEvent(
                category="liquidity",
                name="RRP total z-score",
                severity="medium",
                message=f"RRP total z-score is {rrp_z_latest:.2f} (threshold {RRP_Z_THRESHOLD:.2f}).",
            )
        )

    if "repo_total_z" in signals:
        repo_z_latest = latest_valid(signals["repo_total_z"])
        if repo_z_latest is not None and (
            repo_z_latest >= REPO_Z_THRESHOLD + 0.50
            or last_n_above(signals["repo_total_z"], REPO_Z_THRESHOLD, 3)
        ):
            alerts.append(
                AlertEvent(
                    category="liquidity",
                    name="Repo total z-score",
                    severity="medium",
                    message=f"Repo total z-score is {repo_z_latest:.2f} (threshold {REPO_Z_THRESHOLD:.2f}).",
                )
            )

    # ----------------------------
    # System / regime alert
    # ----------------------------
    if regime != "NORMAL":
        alerts.append(
            AlertEvent(
                category="system",
                name="Market Stress Regime",
                severity="high" if regime == "CRISIS" else "medium",
                message=f"System stress regime = {regime} (prob={stress_prob_latest:.2f}).",
            )
        )

    return alerts, signals


# =========================
# Charting
# =========================


def make_line_chart(
    s: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path,
    hline: Optional[float] = None,
) -> Path:
    s = s.dropna()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(s.index, s.values)
    if hline is not None:
        ax.axhline(hline, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_charts(signals: Dict[str, pd.Series], out_dir: Path) -> List[Path]:
    paths: List[Path] = []

    paths.append(
        make_line_chart(
            last_n_days(signals["hy_oas"], CHART_LOOKBACK_DAYS),
            "Credit Stress: High Yield OAS",
            "Percent",
            out_dir / "hy_oas.png",
            hline=HY_OAS_ABS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            last_n_days(signals["bbb_oas"], CHART_LOOKBACK_DAYS),
            "Credit Stress: BBB OAS",
            "Percent",
            out_dir / "bbb_oas.png",
            hline=BBB_OAS_ABS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            last_n_days(signals["sofr_effr_spread_bps"], CHART_LOOKBACK_DAYS),
            "Liquidity Stress: SOFR - EFFR Spread",
            "Basis Points",
            out_dir / "sofr_effr_spread.png",
            hline=SOFR_EFFR_SPREAD_BPS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            last_n_days(signals["rrp_total"], CHART_LOOKBACK_DAYS),
            "Liquidity Stress: RRP Total",
            "Amount",
            out_dir / "rrp_total.png",
        )
    )

    paths.append(
        make_line_chart(
            last_n_days(signals["hy_minus_bbb"], CHART_LOOKBACK_DAYS),
            "Credit Stress: HY - BBB Spread (Dispersion)",
            "Percent",
            out_dir / "hy_minus_bbb.png",
        )
    )
    if "repo_total" in signals:
        paths.append(
            make_line_chart(
                last_n_days(signals["repo_total"], CHART_LOOKBACK_DAYS),
                "Liquidity Stress: Repo Total",
                "Amount",
                out_dir / "repo_total.png",
            )
        )

    if "stress_score" in signals:
        paths.append(
            make_line_chart(
                last_n_days(signals["stress_score"], CHART_LOOKBACK_DAYS),
                "System Stress: Composite Stress Score",
                "Score",
                out_dir / "stress_score.png",
            )
        )

    if "stress_prob" in signals:
        paths.append(
            make_line_chart(
                last_n_days(signals["stress_prob"] * 100.0, CHART_LOOKBACK_DAYS),
                "System Stress: Stress Probability",
                "Percent",
                out_dir / "stress_probability.png",
                hline=75.0,
            )
        )
    return paths


# =========================
# Email
# =========================


def format_latest_signal_text(
    signals: Dict[str, pd.Series],
    name: str,
    fmt: str = "{:.2f}",
) -> str:
    val = latest_valid(signals[name]) if name in signals else None
    d = latest_date(signals[name]) if name in signals else None
    if val is None or d is None:
        return "n/a"
    return f"{fmt.format(val)} on {d.date()}"


def get_signal_explanations() -> Dict[str, str]:
    return {
        "hy_oas": (
            "High Yield OAS shows how much extra yield lower-quality corporate debt "
            "is paying over safer bonds. Higher values usually mean investors are "
            "getting more worried about credit risk."
        ),
        "bbb_oas": (
            "BBB OAS tracks extra yield for the lower end of investment-grade debt. "
            "If this rises, financing conditions are getting tighter even for better-quality borrowers."
        ),
        "hy_minus_bbb": (
            "HY minus BBB shows the gap between weaker and stronger corporate borrowers. "
            "When this gap widens, markets are becoming more selective and more concerned about risky credit."
        ),
        "sofr_effr_spread_bps": (
            "SOFR minus EFFR is a short-term funding stress signal. "
            "A bigger spread can mean stress in market plumbing or tighter liquidity conditions."
        ),
        "rrp_total": (
            "RRP total shows usage of the Fed's reverse repo facility. "
            "It is more of a liquidity context signal than a direct crisis signal by itself."
        ),
        "repo_total": (
            "Repo total reflects secured short-term funding activity. "
            "Large unusual moves can indicate stress in funding markets or demand for cash."
        ),
        "stress_score": (
            "Stress score is a combined signal built from credit and liquidity indicators. "
            "Higher values mean more overall market stress."
        ),
        "stress_prob": (
            "Stress probability converts the combined stress score into a 0 to 1 scale. "
            "Higher values mean a greater chance that markets are in a stressed regime."
        ),
        "stress_regime": (
            "Stress regime is the overall label for current conditions: NORMAL, WATCH, STRESS, or CRISIS."
        ),
    }


def get_chart_explanations() -> Dict[str, str]:
    return {
        "hy_oas": (
            "This chart shows extra yield demanded for riskier corporate debt. "
            "A rise usually means credit markets are getting nervous."
        ),
        "bbb_oas": (
            "This chart shows extra yield for BBB-rated debt. "
            "A rise suggests tighter borrowing conditions for investment-grade companies."
        ),
        "hy_minus_bbb": (
            "This chart shows the gap between high-yield and BBB spreads. "
            "A widening gap often means weaker borrowers are coming under more pressure."
        ),
        "sofr_effr_spread": (
            "This chart shows short-term funding stress. "
            "Spikes can signal strain in the market's plumbing even before broader risk markets react."
        ),
        "rrp_total": (
            "This chart shows reverse repo facility usage. "
            "It helps describe liquidity conditions, though it should be interpreted with other signals."
        ),
        "repo_total": (
            "This chart shows repo market activity. "
            "Unusual jumps may point to funding demand or liquidity strain."
        ),
        "latent_stress": (
            "This chart shows the model's overall stress level after combining multiple indicators. "
            "Higher values mean more market stress."
        ),
        "stress_probability": (
            "This chart shows the model's estimated probability that markets are in a stressed state. "
            "Higher values mean more caution is warranted."
        ),
        "stress_score": (
            "This chart shows the combined stress score built from credit and liquidity signals. "
            "Higher values mean broader market stress is building."
        ),
    }


def build_snapshot_html(signals: Dict[str, pd.Series]) -> str:
    explanations = get_signal_explanations()

    rows = [
        ("HY OAS", "hy_oas", "{:.2f}%", explanations["hy_oas"]),
        ("BBB OAS", "bbb_oas", "{:.2f}%", explanations["bbb_oas"]),
        ("HY - BBB", "hy_minus_bbb", "{:.2f}%", explanations["hy_minus_bbb"]),
        (
            "SOFR-EFFR spread",
            "sofr_effr_spread_bps",
            "{:.1f} bps",
            explanations["sofr_effr_spread_bps"],
        ),
        ("RRP total", "rrp_total", "{:,.0f}", explanations["rrp_total"]),
    ]

    if "repo_total" in signals:
        rows.append(("Repo total", "repo_total", "{:,.0f}", explanations["repo_total"]))

    if "stress_score" in signals:
        rows.append(
            ("Stress score", "stress_score", "{:.2f}", explanations["stress_score"])
        )

    if "stress_prob" in signals:
        rows.append(
            ("Stress probability", "stress_prob", "{:.1%}", explanations["stress_prob"])
        )

    html_rows = []
    for label, key, fmt, explainer in rows:
        value_text = format_latest_signal_text(signals, key, fmt)
        html_rows.append(
            f"""
            <tr>
              <td style="padding: 8px; border: 1px solid #ddd; vertical-align: top;"><b>{label}</b></td>
              <td style="padding: 8px; border: 1px solid #ddd; vertical-align: top;">{value_text}</td>
              <td style="padding: 8px; border: 1px solid #ddd; vertical-align: top;">{explainer}</td>
            </tr>
            """
        )

    return f"""
    <h3>Snapshot</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 1100px; font-size: 14px;">
      <thead>
        <tr>
          <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Field</th>
          <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Latest</th>
          <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">What it means</th>
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
    """


def build_alerts_html(alerts: List[AlertEvent]) -> str:
    if not alerts:
        return """
        <h3>Alerts</h3>
        <p>No active alerts. This email is just a status snapshot.</p>
        """

    items = []
    for a in alerts:
        items.append(
            f"<li><b>[{a.category.upper()} | {a.severity.upper()}]</b> "
            f"{a.name}: {a.message}</li>"
        )

    return f"""
    <h3>Alerts</h3>
    <p>These are the signals that crossed their warning thresholds.</p>
    <ul>
      {''.join(items)}
    </ul>
    """


def build_regime_html(signals: Dict[str, pd.Series]) -> str:
    regime = "n/a"
    if "stress_regime" in signals:
        latest_regime = signals["stress_regime"].dropna()
        if not latest_regime.empty:
            regime = str(latest_regime.iloc[-1])

    prob_text = (
        format_latest_signal_text(signals, "stress_prob", "{:.1%}")
        if "stress_prob" in signals
        else "n/a"
    )

    return f"""
    <h3>Overall Regime</h3>
    <p>
      <b>Current regime:</b> {regime}<br>
      <b>Stress probability:</b> {prob_text}
    </p>
    <p>
      This is the model's overall read on conditions.
      NORMAL means markets look calm, WATCH means stress is starting to build,
      STRESS means conditions are deteriorating, and CRISIS means risk is elevated.
    </p>
    """


def build_chart_section_html(chart_paths: List[Path]) -> str:
    explanations = get_chart_explanations()
    blocks = []

    for p in chart_paths:
        stem = p.stem
        explainer = explanations.get(
            stem,
            "This chart shows one part of the monitoring system. Higher or unusual moves may indicate building stress.",
        )

        title = stem.replace("_", " ").title()
        blocks.append(
            f"""
            <div style="margin-bottom: 28px;">
              <h3 style="margin-bottom: 8px;">{title}</h3>
              <p style="margin-top: 0; margin-bottom: 10px; line-height: 1.45;">{explainer}</p>
              <img src="cid:{p.name}" style="max-width: 100%; border: 1px solid #ddd;" />
            </div>
            """
        )

    return f"""
    <h3>Charts</h3>
    <p>
      The charts below show how each indicator has been behaving over time.
      What matters most is whether the signal is rising sharply, staying elevated,
      or confirming stress seen in other indicators.
    </p>
    {''.join(blocks)}
    """


def render_alert_html(
    alerts: List[AlertEvent],
    chart_paths: List[Path],
    signals: Dict[str, pd.Series],
) -> str:
    regime_html = build_regime_html(signals)
    snapshot_html = build_snapshot_html(signals)
    alerts_html = build_alerts_html(alerts)
    charts_html = build_chart_section_html(chart_paths)

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #111; line-height: 1.45;">
        <h2>Credit / Liquidity Stress Monitor</h2>

        <p>
          This report tracks signs of pressure in credit markets and short-term funding markets.
          Credit signals help show whether investors are becoming more worried about borrower risk.
          Liquidity signals help show whether market funding conditions are getting strained.
        </p>

        {regime_html}

        {snapshot_html}

        {alerts_html}

        {charts_html}
      </body>
    </html>
    """
    return html


# =========================
# Main
# =========================


def main() -> None:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS)

    # FRED credit stress series
    # HY OAS: BAMLH0A0HYM2
    # BBB OAS: BAMLC0A4CBBB
    hy_oas = fetch_fred_series("BAMLH0A0HYM2", start_date, end_date)
    bbb_oas = fetch_fred_series("BAMLC0A4CBBB", start_date, end_date)

    # NY Fed liquidity stress series using your client
    nyfed = NYFedClient(base_url=NYFED_BASE_URL)
    sofr = fetch_nyfed_series(nyfed, "reference_rates", "SOFR", start_date, end_date)
    effr = fetch_nyfed_series(nyfed, "reference_rates", "EFFR", start_date, end_date)
    rrp_total = fetch_nyfed_series(
        nyfed, "repo_reverse_repo", "RRP_TOTAL", start_date, end_date
    )

    repo_total = None
    if INCLUDE_REPO_TOTAL:
        repo_total = fetch_nyfed_series(
            nyfed, "repo_reverse_repo", "REPO_TOTAL", start_date, end_date
        )

    alerts, signals = analyze_credit_liquidity(
        hy_oas=hy_oas,
        bbb_oas=bbb_oas,
        sofr=sofr,
        effr=effr,
        rrp_total=rrp_total,
        repo_total=repo_total,
        rolling_window=ROLLING_WINDOW,
    )

    if SEND_ONLY_ON_ALERT and not alerts:
        print("No alert triggered. Exiting without email.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        chart_paths = build_charts(signals, out_dir)
        html = render_alert_html(alerts, chart_paths, signals)

        ses = AmazonSES(
            region=SES_REGION,
            access_key=SES_ACCESS_KEY,
            secret_key=SES_SECRET_KEY,
            from_address=SES_FROM_ADDRESS,
        )
        subject_prefix = "ALERT" if alerts else "STATUS"
        ses.send_html_email_many_with_inline_images(
            to_addresses=ALERT_TO,
            subject=f"[{subject_prefix}] Credit / Liquidity Stress Monitor",
            html_content=html,
            image_paths=chart_paths,
        )

    print(f"Email sent. alerts={len(alerts)}")


if __name__ == "__main__":
    main()
