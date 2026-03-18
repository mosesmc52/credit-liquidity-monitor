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
from helpers import str2bool
from nyfed_client import FetchSpec, NYFedClient
from SES import AmazonSES

load_dotenv(find_dotenv())


fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))


# =========================
# Configuration
# =========================

FRED_API_KEY = os.environ["FRED_API_KEY"]

NYFED_BASE_URL = os.getenv("NYFED_BASE_URL", "https://markets.newyorkfed.org/api")

SES_REGION = os.environ["AWS_SES_REGION_NAME"]
SES_ACCESS_KEY = os.environ["AWS_SES_ACCESS_KEY_ID"]
SES_SECRET_KEY = os.environ["AWS_SES_SECRET_ACCESS_KEY"]
SES_FROM_ADDRESS = os.environ["FROM_ADDRESS"]
ALERT_TO = [x.strip() for x in os.environ["TO_ADDRESSES"].split(",") if x.strip()]

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "60"))

# Credit thresholds
HY_OAS_ABS_THRESHOLD = float(os.getenv("HY_OAS_ABS_THRESHOLD", "6.0"))  # percent
BBB_OAS_ABS_THRESHOLD = float(os.getenv("BBB_OAS_ABS_THRESHOLD", "2.5"))  # percent
OAS_Z_THRESHOLD = float(os.getenv("OAS_Z_THRESHOLD", "2.0"))

# Liquidity thresholds
SOFR_EFFR_SPREAD_BPS_THRESHOLD = float(
    os.getenv("SOFR_EFFR_SPREAD_BPS_THRESHOLD", "10.0")
)
RRP_Z_THRESHOLD = float(os.getenv("RRP_Z_THRESHOLD", "2.0"))
REPO_Z_THRESHOLD = float(os.getenv("REPO_Z_THRESHOLD", "2.0"))

SEND_ONLY_ON_ALERT = str2bool(os.getenv("SEND_ONLY_ON_ALERT", True))
INCLUDE_REPO_TOTAL = str2bool(os.getenv("INCLUDE_REPO_TOTAL", False))


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


# =========================
# Analytics
# =========================


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sigma = s.rolling(window).std(ddof=0)
    z = (s - mu) / sigma.replace(0.0, pd.NA)
    return z


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


@dataclass
class AlertEvent:
    category: str
    name: str
    severity: str
    message: str


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

    # Credit
    signals["hy_oas"] = hy_oas
    signals["bbb_oas"] = bbb_oas
    signals["hy_oas_z"] = rolling_zscore(hy_oas, rolling_window)
    signals["bbb_oas_z"] = rolling_zscore(bbb_oas, rolling_window)

    # Credit dispersion (VERY important signal)
    signals["hy_minus_bbb"] = signals["hy_oas"] - signals["bbb_oas"]
    signals["hy_minus_bbb_z"] = rolling_zscore(signals["hy_minus_bbb"], rolling_window)

    # Liquidity
    rates = pd.concat(
        [sofr.rename("SOFR"), effr.rename("EFFR")], axis=1, sort=False
    ).sort_index()
    rates = rates.ffill()
    signals["sofr"] = rates["SOFR"]
    signals["effr"] = rates["EFFR"]
    signals["sofr_effr_spread_bps"] = (rates["SOFR"] - rates["EFFR"]) * 100.0

    signals["rrp_total"] = rrp_total
    signals["rrp_total_z"] = rolling_zscore(rrp_total, rolling_window)

    if repo_total is not None and not repo_total.empty:
        signals["repo_total"] = repo_total
        signals["repo_total_z"] = rolling_zscore(repo_total, rolling_window)

    alerts: List[AlertEvent] = []

    # -------- Credit alerts --------
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
    if hy_z_latest is not None and hy_z_latest >= OAS_Z_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="credit",
                name="High Yield OAS z-score",
                severity="medium",
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
    if bbb_z_latest is not None and bbb_z_latest >= OAS_Z_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="credit",
                name="BBB OAS z-score",
                severity="medium",
                message=f"BBB OAS z-score is {bbb_z_latest:.2f} (threshold {OAS_Z_THRESHOLD:.2f}).",
            )
        )

    # -------- Liquidity alerts --------
    spread_latest = latest_valid(signals["sofr_effr_spread_bps"])
    if spread_latest is not None and spread_latest >= SOFR_EFFR_SPREAD_BPS_THRESHOLD:
        alerts.append(
            AlertEvent(
                category="liquidity",
                name="SOFR-EFFR spread",
                severity="high" if spread_latest >= 20 else "medium",
                message=(
                    f"SOFR-EFFR spread is {spread_latest:.1f} bps "
                    f"(threshold {SOFR_EFFR_SPREAD_BPS_THRESHOLD:.1f} bps)."
                ),
            )
        )

    rrp_z_latest = latest_valid(signals["rrp_total_z"])
    if rrp_z_latest is not None and rrp_z_latest >= RRP_Z_THRESHOLD:
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
        if repo_z_latest is not None and repo_z_latest >= REPO_Z_THRESHOLD:
            alerts.append(
                AlertEvent(
                    category="liquidity",
                    name="Repo total z-score",
                    severity="medium",
                    message=f"Repo total z-score is {repo_z_latest:.2f} (threshold {REPO_Z_THRESHOLD:.2f}).",
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
            signals["hy_oas"],
            "Credit Stress: High Yield OAS",
            "Percent",
            out_dir / "hy_oas.png",
            hline=HY_OAS_ABS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            signals["bbb_oas"],
            "Credit Stress: BBB OAS",
            "Percent",
            out_dir / "bbb_oas.png",
            hline=BBB_OAS_ABS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            signals["sofr_effr_spread_bps"],
            "Liquidity Stress: SOFR - EFFR Spread",
            "Basis Points",
            out_dir / "sofr_effr_spread.png",
            hline=SOFR_EFFR_SPREAD_BPS_THRESHOLD,
        )
    )
    paths.append(
        make_line_chart(
            signals["rrp_total"],
            "Liquidity Stress: RRP Total",
            "Amount",
            out_dir / "rrp_total.png",
        )
    )

    if "repo_total" in signals:
        paths.append(
            make_line_chart(
                signals["repo_total"],
                "Liquidity Stress: Repo Total",
                "Amount",
                out_dir / "repo_total.png",
            )
        )

    return paths


# =========================
# Email
# =========================


def render_alert_html(
    alerts: List[AlertEvent],
    chart_paths: List[Path],
    signals: Dict[str, pd.Series],
) -> str:
    def last_text(name: str, fmt: str = "{:.2f}") -> str:
        val = latest_valid(signals[name])
        d = latest_date(signals[name])
        if val is None or d is None:
            return "n/a"
        return f"{fmt.format(val)} on {d.date()}"

    alert_items = (
        "".join(
            f"<li><b>[{a.category.upper()} | {a.severity.upper()}]</b> {a.name}: {a.message}</li>"
            for a in alerts
        )
        or "<li>No active alerts. Snapshot only.</li>"
    )

    chart_imgs = "".join(
        f"""
        <div style="margin-bottom: 24px;">
          <h3 style="font-family: Arial, sans-serif;">{p.stem.replace('_', ' ').title()}</h3>
          <img src="cid:{p.name}" style="max-width: 100%; border: 1px solid #ddd;" />
        </div>
        """
        for p in chart_paths
    )

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #111;">
        <h2>Credit / Liquidity Stress Monitor</h2>

        <p><b>Snapshot</b></p>
        <ul>
          <li>HY OAS: {last_text("hy_oas", "{:.2f}%")}</li>
          <li>BBB OAS: {last_text("bbb_oas", "{:.2f}%")}</li>
          <li>SOFR-EFFR spread: {last_text("sofr_effr_spread_bps", "{:.1f} bps")}</li>
          <li>RRP total: {last_text("rrp_total", "{:,.0f}")}</li>
        </ul>

        <p><b>Alerts</b></p>
        <ul>{alert_items}</ul>

        {chart_imgs}
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
