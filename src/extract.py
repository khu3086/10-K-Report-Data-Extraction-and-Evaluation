"""End-to-end extraction CLI: PDFs -> located regions -> Claude -> CSV.

Usage:
    python src/extract.py --cycle 1
    python src/extract.py --cycle 2 --reports data/reports --out output

Writes output/extractions_cycle{N}.csv with columns:
    company, ticker, field, key, value, unit, source_page

The hybrid pipeline per (report, field):
    1. parse the PDF once (parse_pdf)
    2. locate the best region for the field (locate)
    3. send that region to Claude for structured extraction (extract_field)
    4. normalize keys and write rows
"""

import argparse
import csv
import glob
import os
from typing import Dict, List

import yaml

from fields import FIELDS, CYCLE_HINTS, normalize_name
from parse import parse_pdf
from locate import locate
from llm import extract_field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_companies(path: str) -> Dict[str, dict]:
    """Map ticker -> company dict from companies.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {c["ticker"]: c for c in data["companies"]}


def _ticker_from_filename(fname: str) -> str:
    """Reports are named <TICKER>.pdf (see edgar_fetch.py)."""
    return os.path.splitext(os.path.basename(fname))[0].upper()


def run(cycle: int, reports_dir: str, out_dir: str, companies_path: str) -> str:
    companies = _load_companies(companies_path)
    cycle_hint = CYCLE_HINTS[cycle]

    pdfs = sorted(glob.glob(os.path.join(reports_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(
            f"No PDFs in {reports_dir}. Run src/edgar_fetch.py first."
        )

    rows: List[dict] = []
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
            result = extract_field(fld, region.context, cycle_hint)
            if result is None:
                continue
            unit = result.detected_scale
            for item in result.items:
                if item.value is None:  # model couldn't find this value
                    continue
                key = "value" if fld.shape == "scalar" else normalize_name(item.key)
                rows.append({
                    "company": company,
                    "ticker": ticker,
                    "field": fld.key,
                    "key": key,
                    "value": item.value,
                    "unit": unit,
                    "source_page": region.page_number,
                })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"extractions_cycle{cycle}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["company", "ticker", "field", "key", "value", "unit", "source_page"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract 10-K fields with the hybrid pipeline.")
    ap.add_argument("--cycle", type=int, default=1, choices=sorted(CYCLE_HINTS))
    ap.add_argument("--reports", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--companies", default=os.path.join(ROOT, "companies.yaml"))
    args = ap.parse_args()
    run(args.cycle, args.reports, args.out, args.companies)


if __name__ == "__main__":
    main()
