"""End-to-end extraction CLI: PDFs -> located regions -> LLM -> CSVs.

Usage:
    python src/extract.py                 # writes BOTH cycle CSVs in one pass
    python src/extract.py --reports data/reports --out output

Each field is extracted from each filing exactly ONCE (a single stable prompt;
see fields.EXTRACT_HINT). Both refinement cycles are then derived from that one
extraction by deterministic post-processing — so the only thing that differs
between cycle 1 and cycle 2 is OUR code, never the LLM's output:

    cycle 1 (baseline): value used as printed (no unit scaling); totals kept.
    cycle 2 (refined):  deterministic unit scaling to actual dollars + totals/
                        reconciliation rows excluded + names normalized.

This isolates the iteration cleanly (cycle 2 == cycle 1 + fixes) and halves the
number of LLM calls.

Writes output/extractions_cycle{1,2}.csv with columns:
    company, ticker, field, key, value, unit, source_page
"""

import argparse
import csv
import glob
import os
from typing import Dict, List

import yaml

from fields import (FIELDS, EXTRACT_HINT, normalize_name, apply_scale,
                    is_total_key, detect_scale)
from parse import parse_pdf
from locate import locate
from llm import extract_field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FIELDNAMES = ["company", "ticker", "field", "key", "value", "unit", "source_page"]


def _load_companies(path: str) -> Dict[str, dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return {c["ticker"]: c for c in data["companies"]}


def _ticker_from_filename(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0].upper()


def _rows_for_cycles(company, ticker, fld, result, page, scale):
    """Turn one extraction into (cycle1_rows, cycle2_rows) via post-processing.

    `scale` is the deterministically-detected table scale (falls back to the
    LLM's if detection found nothing).
    """
    c1: List[dict] = []
    c2: List[dict] = []
    for item in result.items:
        if item.value is None:
            continue
        key = "value" if fld.shape == "scalar" else normalize_name(item.key)
        base = {"company": company, "ticker": ticker, "field": fld.key,
                "key": key, "source_page": page}

        # Cycle 1 — naive: value as printed, totals included.
        c1.append({**base, "value": item.value, "unit": scale})

        # Cycle 2 — refined: scale to actual dollars, drop totals/reconciliations.
        if fld.shape == "keyed" and is_total_key(key):
            continue
        c2.append({**base, "value": apply_scale(item.value, scale), "unit": "actual"})
    return c1, c2


def run(reports_dir: str, out_dir: str, companies_path: str) -> None:
    companies = _load_companies(companies_path)
    pdfs = sorted(glob.glob(os.path.join(reports_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs in {reports_dir}. Run src/edgar_fetch.py first.")

    rows = {1: [], 2: []}
    for pdf in pdfs:
        ticker = _ticker_from_filename(pdf)
        company = companies.get(ticker, {}).get("name", ticker)
        print(f"[{ticker}] parsing {os.path.basename(pdf)} ...")
        pages = parse_pdf(pdf)

        for fld in FIELDS:
            region = locate(fld, pages)
            if region is None:
                print(f"  [{fld.key}] no anchor matched — skipping")
                continue
            print(f"  [{fld.key}] region p{region.page_number} (score {region.score})")
            result = extract_field(fld, region.context, EXTRACT_HINT)
            if result is None:
                continue
            # Deterministic scale from the page text; fall back to the LLM's.
            scale = detect_scale(region.context) or result.detected_scale
            c1, c2 = _rows_for_cycles(company, ticker, fld, result, region.page_number, scale)
            rows[1].extend(c1)
            rows[2].extend(c2)

    os.makedirs(out_dir, exist_ok=True)
    for cycle in (1, 2):
        out_path = os.path.join(out_dir, f"extractions_cycle{cycle}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows[cycle])
        print(f"Wrote {len(rows[cycle])} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract 10-K fields; emit both cycle CSVs.")
    ap.add_argument("--reports", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--companies", default=os.path.join(ROOT, "companies.yaml"))
    # Accepted for backward compatibility; extraction always emits both cycles.
    ap.add_argument("--cycle", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    run(args.reports, args.out, args.companies)


if __name__ == "__main__":
    main()
