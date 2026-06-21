"""Build an INDEPENDENT ground-truth dataset (no manual entry).

Two independent sources, neither of which is the system under test
(gpt-4o-mini):

  - rd_expense        -> SEC XBRL Company Concept API (authoritative, structured,
                         deterministic — the company's own tagged number).
  - segment_revenue   -> a STRONGER independent model (default openai/gpt-4o) run
    geographic_revenue   over the located region, with deterministic Python
                         scaling. This is "silver" ground truth for the two
                         dimensionally-tagged fields XBRL doesn't expose cleanly.

Matches whatever 10-K was actually fetched (the latest), by taking the most
recent annual XBRL fact rather than a hard-coded fiscal year.

Usage:
    python src/build_ground_truth.py
    GT_MODEL=openai/gpt-4o python src/build_ground_truth.py --only AAPL,MSFT
"""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import List, Optional

import requests
import yaml

from fields import (FIELDS_BY_KEY, EXTRACT_HINT, normalize_name, apply_scale,
                    is_total_key, detect_scale)
from parse import parse_pdf
from locate import locate
from llm import extract_field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GT_MODEL = os.environ.get("GT_MODEL", "openai/gpt-4o")
USER_AGENT = os.environ.get("SEC_USER_AGENT", "10-K Extraction Project lokesh.nigam@gmail.com")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
XBRL_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

# us-gaap tags companies use for R&D (first match wins).
RD_TAGS = [
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
]


def _load_companies(path: str) -> List[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["companies"]


def _is_annual(fact: dict) -> bool:
    """True if the fact covers ~one year (a full fiscal-year duration)."""
    try:
        s = datetime.strptime(fact["start"], "%Y-%m-%d")
        e = datetime.strptime(fact["end"], "%Y-%m-%d")
        return 350 <= (e - s).days <= 380
    except (KeyError, ValueError):
        return False


def rd_from_xbrl(cik: int) -> Optional[float]:
    """Most recent annual R&D expense (actual dollars) from SEC XBRL, or None.

    Companies tag R&D inconsistently: some use ResearchAndDevelopmentExpense as
    the main line, others (e.g. J&J) put the headline figure under
    ...ExcludingAcquiredInProcessCost and use ResearchAndDevelopmentExpense for a
    small in-process sub-line. So we collect the latest annual value from every
    candidate tag and, among those sharing the latest fiscal-year end, take the
    LARGEST (the main R&D line; sub-components are always smaller).
    """
    candidates: List[tuple] = []  # (end_date, value)
    for tag in RD_TAGS:
        url = XBRL_URL.format(cik=cik, tag=tag)
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        annual = [
            f for f in r.json().get("units", {}).get("USD", [])
            if str(f.get("form", "")).startswith("10-K")
            and f.get("fp") == "FY"
            and _is_annual(f)
        ]
        if annual:
            best = max(annual, key=lambda f: f["end"])  # latest fiscal year
            candidates.append((best["end"], float(best["val"])))
    if not candidates:
        return None
    latest_end = max(end for end, _ in candidates)
    return max(v for end, v in candidates if end == latest_end)


def seg_geo_from_model(pdf_path: str) -> List[dict]:
    """Segment + geographic revenue via a stronger independent model.

    Reuses the same locate step, the cycle-2 prompt, and deterministic Python
    scaling — but with GT_MODEL (default gpt-4o), independent of the gpt-4o-mini
    system under test.
    """
    pages = parse_pdf(pdf_path)
    out: List[dict] = []
    for fkey in ("segment_revenue", "geographic_revenue"):
        fld = FIELDS_BY_KEY[fkey]
        region = locate(fld, pages)
        if region is None:
            continue
        res = extract_field(fld, region.context, EXTRACT_HINT, model=GT_MODEL)
        if res is None:
            continue
        scale = detect_scale(region.context) or res.detected_scale  # same as cycle 2
        for item in res.items:
            if item.value is None:
                continue
            key = normalize_name(item.key)
            if is_total_key(key):  # mirror cycle-2 post-processing
                continue
            out.append({
                "field": fkey,
                "key": key,
                "value": apply_scale(item.value, scale),
                "source_page": region.page_number,
            })
    return out


def run(companies_path: str, reports_dir: str, out_path: str,
        only: Optional[List[str]]) -> None:
    companies = _load_companies(companies_path)
    if only:
        wanted = {t.upper() for t in only}
        companies = [c for c in companies if c["ticker"].upper() in wanted]

    rows: List[dict] = []
    for c in companies:
        ticker, cik, name = c["ticker"], int(c["cik"]), c["name"]
        pdf = os.path.join(reports_dir, f"{ticker}.pdf")
        if not os.path.exists(pdf):
            print(f"[{ticker}] no PDF — skipping")
            continue
        print(f"[{ticker}] R&D from XBRL ...")
        rd = rd_from_xbrl(cik)
        time.sleep(0.3)
        if rd is not None:
            rows.append({"company": name, "ticker": ticker, "field": "rd_expense",
                         "key": "value", "value": rd, "unit": "actual", "source_page": ""})
            print(f"  rd_expense = {rd:,.0f}")
        else:
            print(f"  rd_expense not tagged in XBRL — omitted")

        print(f"[{ticker}] segment + geographic from {GT_MODEL} ...")
        for r in seg_geo_from_model(pdf):
            rows.append({"company": name, "ticker": ticker, "field": r["field"],
                         "key": r["key"], "value": r["value"], "unit": "actual",
                         "source_page": r["source_page"]})
        print(f"  +{sum(1 for x in rows if x['ticker'] == ticker) } rows total for {ticker}")

    cols = ["company", "ticker", "field", "key", "value", "unit", "source_page"]

    # When regenerating a subset, preserve every other company's existing rows
    # instead of overwriting the whole file (free-tier rate limits make full
    # rebuilds expensive). Provenance comment lines (leading '#') are dropped and
    # re-emitted fresh.
    # Replace at (ticker, field) granularity; a field that yielded nothing this
    # run (XBRL missing or model failed) keeps its existing rows instead of being
    # wiped by a partial regeneration.
    preserved: List[dict] = []
    if only and os.path.exists(out_path):
        refreshed = {(r["ticker"].upper(), r["field"]) for r in rows}
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("company") or "").startswith("#"):
                    continue
                if ((r.get("ticker") or "").upper(), r.get("field")) not in refreshed:
                    preserved.append({k: r.get(k, "") for k in cols})

    all_rows = preserved + rows
    all_rows.sort(key=lambda r: (r["ticker"], r["field"], str(r["key"])))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        # Provenance note (skipped by evaluate.py's '#'-prefix filter).
        w.writerow({"company": "# rd_expense = SEC XBRL (authoritative); "
                    "segment/geographic = " + GT_MODEL + " (independent silver GT)",
                    "ticker": "", "field": "", "key": "", "value": "", "unit": "",
                    "source_page": ""})
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} ground-truth rows -> {out_path} "
          f"({len(rows)} regenerated, {len(preserved)} preserved)")


def main():
    ap = argparse.ArgumentParser(description="Build independent ground truth (XBRL + strong model).")
    ap.add_argument("--companies", default=os.path.join(ROOT, "companies.yaml"))
    ap.add_argument("--reports", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ground_truth.csv"))
    ap.add_argument("--only", default=None, help="Comma-separated tickers")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    run(args.companies, args.reports, args.out, only)


if __name__ == "__main__":
    main()
