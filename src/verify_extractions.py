"""Offline sanity check: do extracted segment values actually appear in the PDF?

For each company we take its segment_revenue values (actual $), render them the
way a 10-K prints them (whole millions, with and without thousands commas), and
check whether that exact figure is present in the filing text. Fabricated or
broken numbers (round 100/80/60B placeholders, or all-zero rows) won't be found;
genuine extractions will.

This needs NO API calls — it's a cheap, deterministic guardrail that flags which
companies' extractions are trustworthy before (or independently of) the scored
evaluation against ground truth.

Usage:
    python src/verify_extractions.py            # cycle 2
    python src/verify_extractions.py --cycle 1
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

from pdfminer.high_level import extract_text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def present(millions: int, text: str) -> bool:
    """True only if the figure appears as a real, boundaried number in the text.

    Checks the comma-grouped form (178,353) and the bare form with non-digit
    boundaries (so 20000 doesn't match inside 1,200,000). Avoids the substring
    false positives that make round placeholders look real.
    """
    if not millions:
        return False
    for fm in {f"{millions:,}", str(millions)}:
        if re.search(r"(?<!\d)" + re.escape(fm) + r"(?!\d)", text):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", type=int, default=2)
    ap.add_argument("--field", default="segment_revenue")
    ap.add_argument("--reports", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    args = ap.parse_args()

    csv_path = os.path.join(args.out, f"extractions_cycle{args.cycle}.csv")
    rows = list(csv.DictReader(open(csv_path)))
    by_tkr = defaultdict(list)
    for r in rows:
        if r["field"] == args.field:
            by_tkr[r["ticker"]].append((r["key"], float(r["value"])))

    print(f"Verifying {args.field}, cycle {args.cycle} — value present in source PDF?\n")
    print(f"{'TICKER':7} {'found/total':12} verdict")
    print("-" * 64)
    th = tt = 0
    suspect = []
    for tkr in sorted(by_tkr):
        pdf = os.path.join(args.reports, f"{tkr}.pdf")
        if not os.path.exists(pdf):
            continue
        text = extract_text(pdf)
        hits, miss = 0, []
        for key, val in by_tkr[tkr]:
            if present(round(val / 1e6), text):
                hits += 1
            else:
                miss.append(f"{key}={val/1e9:.1f}B")
        total = len(by_tkr[tkr])
        th += hits
        tt += total
        rate = hits / total if total else 0
        verdict = "OK" if rate >= 0.7 else ("PARTIAL" if rate > 0 else "FABRICATED/BROKEN")
        if rate < 0.7:
            suspect.append(tkr)
        extra = ("  missing: " + ", ".join(miss)) if miss else ""
        print(f"{tkr:7} {f'{hits}/{total}':12} {verdict}{extra}")

    print("-" * 64)
    pct = (th / tt) if tt else 0
    print(f"TOTAL cells found in source PDFs: {th}/{tt} ({pct:.0%})")
    if suspect:
        print(f"Needs regeneration: {', '.join(suspect)}")
    return 1 if suspect else 0


if __name__ == "__main__":
    sys.exit(main())
