# nyfed_client.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass
class FetchSpec:
    dataset: str
    key: str


class NYFedClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def fetch_series(
        self,
        spec: FetchSpec,
        start_date: dt.date,
        end_date: dt.date,
        timeout: int = 30,
    ) -> List[Tuple[dt.date, float]]:
        dataset = spec.dataset.strip().lower()
        if dataset in {"reference_rates", "rates"}:
            return self._fetch_reference_rates(spec, start_date, end_date, timeout)
        if dataset in {"repo_reverse_repo", "rp"}:
            return self._fetch_repo_reverse_repo(spec, start_date, end_date, timeout)
        if dataset in {"central_bank_liquidity_swaps", "cbls"}:
            return self._fetch_cbls(spec, start_date, end_date, timeout)

        raise ValueError(
            f"Unsupported NY Fed dataset '{spec.dataset}'. "
            "Supported: reference_rates, repo_reverse_repo, central_bank_liquidity_swaps."
        )

    def _fetch_reference_rates(
        self,
        spec: FetchSpec,
        start_date: dt.date,
        end_date: dt.date,
        timeout: int,
    ) -> List[Tuple[dt.date, float]]:
        # Official endpoint family: /api/rates/...
        search_url = f"{self.base_url}/rates/all/search.json"
        params = {"startDate": str(start_date), "endDate": str(end_date)}
        rows: List[Dict[str, Any]] = []

        try:
            r = requests.get(search_url, params=params, timeout=timeout)
            r.raise_for_status()
            rows = r.json().get("refRates", [])
        except requests.RequestException:
            latest_url = f"{self.base_url}/rates/all/latest.json"
            r = requests.get(latest_url, timeout=timeout)
            r.raise_for_status()
            rows = r.json().get("refRates", [])

        out: List[Tuple[dt.date, float]] = []
        target = spec.key.strip().upper()
        for row in rows:
            row_type = str(row.get("type", "")).upper()
            if target and target != "ALL" and row_type != target:
                continue

            d = self._coerce_date(row, "effectiveDate", "date")
            if d is None or d < start_date or d > end_date:
                continue

            v = self._coerce_float(
                row,
                "percentRate",
                "value",
                "index",
                "average30day",
                "average90day",
                "average180day",
            )
            if v is None:
                continue
            out.append((d, v))

        out.sort(key=lambda x: x[0])
        return out

    def _fetch_repo_reverse_repo(
        self,
        spec: FetchSpec,
        start_date: dt.date,
        end_date: dt.date,
        timeout: int,
    ) -> List[Tuple[dt.date, float]]:
        # Official endpoint family: /api/rp/...
        # Prefer documented rpops search, then fall back to historical combined path.
        payload = None
        last_err: Optional[Exception] = None
        candidates = [
            (
                f"{self.base_url}/rp/rpops/search.json",
                {"startDate": str(start_date), "endDate": str(end_date)},
            ),
            (f"{self.base_url}/rp/rpops/lastTwoWeeks.json", None),
            (f"{self.base_url}/rp/all/all/results/lastTwoWeeks.json", None),
        ]
        for url, params in candidates:
            try:
                r = requests.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                payload = r.json()
                break
            except requests.RequestException as exc:
                last_err = exc

        if payload is None:
            if last_err is not None:
                raise last_err
            return []
        data = payload

        target = spec.key.strip().upper()
        include_repo = target in {"", "ALL", "REPO_TOTAL", "REPO_TOTAL_ACCEPTED"}
        include_rrp = target in {
            "ALL",
            "RRP_TOTAL",
            "RRP_TOTAL_ACCEPTED",
            "REVERSE_REPO_TOTAL",
        }

        repo_rows = (
            data.get("repo", {}).get("operations", [])
            if isinstance(data.get("repo"), dict)
            else []
        )
        rrp_rows = (
            data.get("reverseRepo", {}).get("operations", [])
            if isinstance(data.get("reverseRepo"), dict)
            else []
        )

        totals_by_date: Dict[dt.date, float] = {}
        if include_repo:
            self._accumulate_ops_by_date(
                totals_by_date, repo_rows, start_date, end_date
            )
        if include_rrp:
            self._accumulate_ops_by_date(totals_by_date, rrp_rows, start_date, end_date)

        out = sorted(totals_by_date.items(), key=lambda x: x[0])
        return out

    def _fetch_cbls(
        self,
        spec: FetchSpec,
        start_date: dt.date,
        end_date: dt.date,
        timeout: int,
    ) -> List[Tuple[dt.date, float]]:
        # In NY Fed Markets API this product is published under /api/fxs/...
        # (foreign exchange liquidity swaps).
        candidates: List[Tuple[str, Optional[Dict[str, str]]]] = [
            (
                f"{self.base_url}/fxs/all/results/search.json",
                {"startDate": str(start_date), "endDate": str(end_date)},
            ),
            (f"{self.base_url}/fxs/usdollar/last/14.json", None),
        ]

        payload: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None
        for url, params in candidates:
            try:
                r = requests.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                payload = r.json()
                break
            except requests.RequestException as exc:
                last_err = exc

        if payload is None:
            if last_err is not None:
                raise last_err
            return []

        rows = self._extract_rows(payload)
        target = spec.key.strip().upper()
        out: List[Tuple[dt.date, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            row_key = str(
                row.get("type")
                or row.get("series")
                or row.get("seriesId")
                or row.get("metric")
                or row.get("counterparty")
                or ""
            ).upper()
            if target and target != "ALL" and row_key and row_key != target:
                continue

            d = self._coerce_date(
                row, "operationDate", "effectiveDate", "date", "asOfDate"
            )
            if d is None or d < start_date or d > end_date:
                continue

            v = self._coerce_float(
                row,
                "value",
                "amount",
                "total",
                "outstanding",
                "usdAmount",
                "dollarAmount",
                "totalAmtAccepted",
                "totalAmtSubmitted",
            )
            if v is None:
                continue
            out.append((d, v))

        out.sort(key=lambda x: x[0])
        return out

    @staticmethod
    def _coerce_date(row: Dict[str, Any], *keys: str) -> Optional[dt.date]:
        for key in keys:
            raw = row.get(key)
            if not raw:
                continue
            txt = str(raw)[:10]
            try:
                return dt.date.fromisoformat(txt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _coerce_float(row: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            raw = row.get(key)
            if raw is None:
                continue
            txt = str(raw).strip()
            if txt in {"", ".", "null", "None"}:
                continue
            txt = txt.replace(",", "")
            try:
                return float(txt)
            except ValueError:
                continue
        return None

    def _accumulate_ops_by_date(
        self,
        totals_by_date: Dict[dt.date, float],
        rows: Iterable[Dict[str, Any]],
        start_date: dt.date,
        end_date: dt.date,
    ) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            d = self._coerce_date(row, "operationDate", "effectiveDate", "date")
            if d is None or d < start_date or d > end_date:
                continue
            v = self._coerce_float(
                row,
                "totalAmtAccepted",
                "totalAcceptedAmt",
                "totalAmtSubmitted",
                "acceptedAmount",
                "amount",
                "value",
            )
            if v is None:
                continue
            totals_by_date[d] = totals_by_date.get(d, 0.0) + v

    def _extract_rows(self, payload: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        def walk(node: Any) -> None:
            if isinstance(node, list):
                if node and all(isinstance(item, dict) for item in node):
                    rows.extend(node)
                for item in node:
                    walk(item)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)

        walk(payload)
        return rows
