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


def _merge_existing(out_path: str, new_rows: List[dict], only: set) -> List[dict]:
    """Combine freshly-extracted rows with the existing CSV.

    Rows for tickers in `only` are replaced by `new_rows`; every other company's
    rows are preserved as-is. This lets us re-extract one or two filings without
    discarding the rest (important under free-tier API rate limits).
    """
    # Replace at (ticker, field) granularity: only the specific field that
    # produced fresh rows is swapped in. A field that failed (e.g. rate-limited)
    # keeps its existing rows rather than being silently wiped — so a partial
    # success never destroys the fields that didn't come back this run.
    refreshed = {(r["ticker"].upper(), r["field"]) for r in new_rows}
    kept: List[dict] = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                if (r["ticker"].upper(), r["field"]) not in refreshed:
                    kept.append({k: r.get(k, "") for k in FIELDNAMES})
    refreshed_tickers = {t for t, _ in refreshed}
    skipped = only - refreshed_tickers
    if skipped:
        print(f"  WARNING: no new rows for {', '.join(sorted(skipped))} — "
              f"keeping existing rows for those.")
    return kept + new_rows


def run(reports_dir: str, out_dir: str, companies_path: str, only=None) -> None:
    companies = _load_companies(companies_path)
    pdfs = sorted(glob.glob(os.path.join(reports_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs in {reports_dir}. Run src/edgar_fetch.py first.")

    only = {t.upper() for t in only} if only else None
    if only:
        pdfs = [p for p in pdfs if _ticker_from_filename(p) in only]
        missing = only - {_ticker_from_filename(p) for p in pdfs}
        if missing:
            raise SystemExit(f"No PDF found for: {', '.join(sorted(missing))}")
        print(f"--only: re-extracting {', '.join(sorted(only))} "
              f"(other companies preserved).")

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
        out_rows = _merge_existing(out_path, rows[cycle], only) if only else rows[cycle]
        # Keep a stable, ticker-grouped order so diffs stay readable.
        out_rows.sort(key=lambda r: (r["ticker"], r["field"], str(r["key"])))
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {len(out_rows)} rows -> {out_path}"
              f" ({len(rows[cycle])} re-extracted)" if only else
              f"Wrote {len(out_rows)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract 10-K fields; emit both cycle CSVs.")
    ap.add_argument("--reports", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--companies", default=os.path.join(ROOT, "companies.yaml"))
    ap.add_argument("--only", default=None,
                    help="Comma-separated tickers to re-extract; other companies "
                         "in the existing CSVs are preserved (merge, not overwrite).")
    # Accepted for backward compatibility; extraction always emits both cycles.
    ap.add_argument("--cycle", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    only = [t.strip() for t in args.only.split(",")] if args.only else None
    run(args.reports, args.out, args.companies, only=only)


if __name__ == "__main__":
    main()
