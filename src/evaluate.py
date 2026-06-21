"""Evaluate extractions against ground truth and emit metrics + an error log.

Metrics:
  - Scalar fields (rd_expense): accuracy = fraction within relative tolerance;
    MAPE over all companies present in both sets.
  - Keyed fields (segment_revenue, geographic_revenue): precision / recall / F1
    over (normalized-key, value-within-tolerance) pairs; value MAPE on matches.

Also writes a per-row error log classifying each miss into a category, which is
what drives the cycle-1 -> cycle-2 error analysis.

Usage:
    python src/evaluate.py --cycle 1
    python src/evaluate.py --cycle 2 --truth data/ground_truth.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from fields import FIELDS_BY_KEY, normalize_name

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOLERANCE = 0.005  # 0.5% relative tolerance absorbs rounding / unit jitter

# (ticker, field, normalized_key) -> value
Cell = Tuple[str, str, str]


def _load(path: str) -> Dict[Cell, float]:
    out: Dict[Cell, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            # Skip blank rows and "# ..." comment lines in the CSV.
            if not row.get("value"):
                continue
            if (row.get("company") or "").lstrip().startswith("#"):
                continue
            try:
                val = float(row["value"])
            except ValueError:
                continue
            field = row["field"]
            key = "value" if FIELDS_BY_KEY.get(field) and FIELDS_BY_KEY[field].shape == "scalar" \
                else normalize_name(row["key"])
            out[(row["ticker"].upper(), field, key)] = val
    return out


def _close(a: float, b: float) -> bool:
    if b == 0:
        return abs(a) < 1.0
    return abs(a - b) / abs(b) <= TOLERANCE


def _classify(pred: float, truth: float) -> str:
    """Categorize a value mismatch to inform error analysis."""
    if truth == 0:
        return "other"
    ratio = pred / truth if truth else 0
    # Scale errors: off by ~1000x or ~1e6 in either direction
    for scale, name in [(1e3, "scale_x1000"), (1e6, "scale_x1e6")]:
        if _close(ratio, scale) or _close(ratio, 1 / scale):
            return name
    # Prior-year column confusion typically lands within ~30% of the truth
    if 0.6 <= ratio <= 0.97 or 1.03 <= ratio <= 1.6:
        return "wrong_year_or_rounding"
    return "wrong_value"


def _group(cells: Dict[Cell, float]) -> Dict[tuple, Dict[str, float]]:
    """Group (ticker, field, key)->value into {(ticker, field): {key: value}}."""
    g: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for (ticker, field, key), v in cells.items():
        g[(ticker, field)][key] = v
    return g


def evaluate(pred_path: str, truth_path: str, out_dir: str, cycle: int) -> dict:
    preds = _load(pred_path)
    truth = _load(truth_path)
    pg, tg = _group(preds), _group(truth)

    per_field: Dict[str, dict] = {}
    errors: List[dict] = []

    for fkey, fld in FIELDS_BY_KEY.items():
        tickers = sorted(
            {t for (t, f) in tg if f == fkey} | {t for (t, f) in pg if f == fkey})

        tp = fp = fn = 0
        ape_sum = 0.0
        ape_n = 0

        # Per-company bipartite matching. A prediction is a true positive if it
        # matches a truth cell by exact (normalized) key OR, failing that, by
        # value within tolerance — so a correct number with a name variant
        # ("greater china" vs "china") still counts. Wrong values and spurious
        # keys are false positives; unmatched truth cells are false negatives.
        for ticker in tickers:
            t_items = dict(tg.get((ticker, fkey), {}))
            p_items = dict(pg.get((ticker, fkey), {}))
            matched_t: set = set()
            used_p: set = set()

            # Pass 1: exact-key correct matches
            for pk, pv in p_items.items():
                if pk in t_items and _close(pv, t_items[pk]):
                    tp += 1
                    used_p.add(pk)
                    matched_t.add(pk)
                    ape_sum += abs(pv - t_items[pk]) / abs(t_items[pk]) if t_items[pk] else 0
                    ape_n += 1

            # Pass 2: value-based matches (name variants)
            for pk, pv in p_items.items():
                if pk in used_p:
                    continue
                cand = next((tk for tk, tv in t_items.items()
                             if tk not in matched_t and _close(pv, tv)), None)
                if cand is not None:
                    tp += 1
                    used_p.add(pk)
                    matched_t.add(cand)
                    ape_sum += abs(pv - t_items[cand]) / abs(t_items[cand]) if t_items[cand] else 0
                    ape_n += 1

            # Remaining predictions -> false positives
            for pk, pv in p_items.items():
                if pk in used_p:
                    continue
                fp += 1
                cat = _classify(pv, t_items[pk]) if pk in t_items else "spurious_key"
                errors.append({"ticker": ticker, "field": fkey, "key": pk,
                               "predicted": pv, "truth": t_items.get(pk, ""),
                               "category": cat})

            # Unmatched truth cells -> false negatives
            for tk, tv in t_items.items():
                if tk not in matched_t:
                    fn += 1
                    cat = _classify(p_items[tk], tv) if tk in p_items else "missing_key"
                    errors.append({"ticker": ticker, "field": fkey, "key": tk,
                                   "predicted": p_items.get(tk, ""), "truth": tv,
                                   "category": cat})

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        mape = (ape_sum / ape_n) if ape_n else None

        per_field[fkey] = {
            "shape": fld.shape,
            "truth_cells": sum(1 for (t, f) in tg if f == fkey for _ in tg[(t, f)]),
            "predicted_cells": sum(1 for (t, f) in pg if f == fkey for _ in pg[(t, f)]),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(recall, 4) if fld.shape == "scalar" else None,
            "value_mape": round(mape, 4) if mape is not None else None,
        }

    # Overall: micro-averaged across fields
    tot_tp = sum(m["tp"] for m in per_field.values())
    tot_fp = sum(m["fp"] for m in per_field.values())
    tot_fn = sum(m["fn"] for m in per_field.values())
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Error-category histogram for the analysis writeup
    cat_counts: Dict[str, int] = defaultdict(int)
    for e in errors:
        cat_counts[e["category"]] += 1

    summary = {
        "cycle": cycle,
        "tolerance": TOLERANCE,
        "overall": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
            "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        },
        "per_field": per_field,
        "error_categories": dict(sorted(cat_counts.items(), key=lambda kv: -kv[1])),
    }

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"metrics_cycle{cycle}.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    errors_path = os.path.join(out_dir, f"errors_cycle{cycle}.csv")
    with open(errors_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ticker", "field", "key", "predicted", "truth", "category"]
        )
        w.writeheader()
        w.writerows(errors)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {metrics_path}\nWrote {errors_path} ({len(errors)} error rows)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate extractions vs ground truth.")
    ap.add_argument("--cycle", type=int, default=1)
    ap.add_argument("--pred", default=None, help="Defaults to output/extractions_cycle{N}.csv")
    ap.add_argument("--truth", default=os.path.join(ROOT, "data", "ground_truth.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    args = ap.parse_args()
    pred = args.pred or os.path.join(args.out, f"extractions_cycle{args.cycle}.csv")
    evaluate(pred, args.truth, args.out, args.cycle)


if __name__ == "__main__":
    main()
