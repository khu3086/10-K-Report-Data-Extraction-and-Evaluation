"""Field definitions, location anchors, and normalization rules.

This is the primary "live-edit" surface for the project. Each field declares:
  - how to *find* its table/section in the 10-K (anchor keywords),
  - what *shape* the answer takes (scalar vs. keyed multi-value),
  - cycle-specific extraction hints that drive the improvement story.

The iteration story lives in `CYCLE_HINTS`: cycle 1 is a naive baseline whose
errors we analyze; cycle 2 adds targeted instructions that fix the dominant
error categories (scale detection, current-vs-prior-year column, segment-name
normalization, geographic-note location).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Field:
    key: str                      # machine id, used in CSVs
    label: str                    # human label
    shape: str                    # "scalar" or "keyed" (multi-value)
    anchors: List[str]            # keyword phrases that mark the relevant region
    description: str              # what to extract (sent to the LLM)


# --- The three extracted fields -------------------------------------------------

FIELDS: List[Field] = [
    Field(
        key="segment_revenue",
        label="Segment revenue",
        shape="keyed",
        anchors=[
            "reportable segments",
            "operating segments",
            "segment information",
            "information by segment",
            "results of operations by segment",
        ],
        description=(
            "Revenue (net sales) for each reportable operating segment for the "
            "most recent fiscal year. Return one entry per segment with the "
            "segment name as the key and its revenue as the value."
        ),
    ),
    Field(
        key="geographic_revenue",
        label="Geographic revenue breakdown",
        shape="keyed",
        anchors=[
            "geographic",
            "by geography",
            "geographical",
            "revenue by region",
            "net sales by",
        ],
        description=(
            "Revenue (net sales) broken down by geography for the most recent "
            "fiscal year (e.g. United States vs. International, or by region). "
            "Return one entry per geographic area with the area name as the key "
            "and its revenue as the value. Use the most granular breakdown that "
            "is consistently presented."
        ),
    ),
    Field(
        key="rd_expense",
        label="Research & development expense",
        shape="scalar",
        anchors=[
            "research and development",
            "research and development expense",
            "research, development",
            "technology and content",  # Amazon's label
        ],
        description=(
            "Total research and development expense for the most recent fiscal "
            "year. Return a single numeric value."
        ),
    ),
]

FIELDS_BY_KEY: Dict[str, Field] = {f.key: f for f in FIELDS}


# --- Per-cycle extraction hints (the improvement story) -------------------------
# Cycle 1: deliberately minimal — expected to make errors worth analyzing.
# Cycle 2: targeted fixes derived from error analysis of cycle 1.

# A single, stable extraction prompt used for ALL cycles. The model just reads
# the table faithfully (number as printed + the table's scale + the current
# fiscal year). The cycle-to-cycle improvement is then applied as deterministic
# POST-PROCESSING in extract.py, so the only thing that changes between cycles is
# our code — not the LLM's output. This isolates the iteration cleanly.
EXTRACT_HINT: str = (
    "Extract the requested values from the provided 10-K excerpt.\n"
    "1. Return each value EXACTLY as printed in the table — do NOT convert units "
    "(e.g. print '34,550' as 34550). Report the table's scale in 'detected_scale'.\n"
    "2. Use the most recent fiscal year (the leftmost / current-year column).\n"
    "3. Include every relevant line, using each label exactly as printed.\n"
    "4. Omit any value not present in the excerpt rather than guessing."
)

# What each cycle's post-processing does (for the dashboard / write-up).
CYCLE_NOTES: Dict[int, str] = {
    1: "Baseline: values used as printed (no unit scaling); totals not removed.",
    2: "Refined: deterministic unit scaling to actual dollars + totals/"
       "reconciliation rows excluded + segment/region names normalized.",
}

# Rows that are aggregates / reconciliations, not a segment or region. Excluded
# in cycle 2 (and in the ground truth) so they don't count as real entries.
TOTAL_PATTERNS = (
    "total", "consolidated", "eliminat", "intersegment", "reconcil",
    "corporate and other", "corporate/other", "all other", "segment total",
)


def is_total_key(key: str) -> bool:
    """True if a normalized key is an aggregate/reconciliation row, not a real one."""
    k = (key or "").strip().lower()
    return any(p in k for p in TOTAL_PATTERNS)


# --- Scale handling (deterministic, used by the cycle-2 pipeline) ---------------

SCALE_FACTORS: Dict[str, float] = {
    "units": 1.0,
    "ones": 1.0,
    "thousands": 1_000.0,
    "millions": 1_000_000.0,
    "billions": 1_000_000_000.0,
}


def scale_factor(detected_scale: str) -> float:
    """Map a detected scale label to a multiplier (defaults to 1.0)."""
    return SCALE_FACTORS.get((detected_scale or "units").strip().lower(), 1.0)


def detect_scale(text: str) -> Optional[str]:
    """Detect a table's unit scale from its surrounding text (deterministic).

    10-K tables state their scale in the header — "(in millions)", "amounts in
    thousands", "$ in billions". Reading it directly from the page is more
    reliable than trusting the LLM, and — crucially — it lets predictions and the
    ground truth use the SAME scale, eliminating cross-model 1000x mismatches.
    Returns None if no scale phrase is found (caller falls back to the LLM's).
    """
    t = (text or "").lower()
    counts = []
    for scale in ("billions", "millions", "thousands"):
        n = len(re.findall(r"in\s+" + scale, t))
        if n:
            counts.append((n, scale))
    if counts:
        counts.sort(reverse=True)  # most frequently stated scale wins
        return counts[0][1]
    return None


def apply_scale(value: float, detected_scale: str) -> float:
    """Scale a printed value to actual dollars, guarding against double-scaling.

    The cycle-2 prompt asks the model for the number *as printed* so Python can
    multiply by the detected scale. But models sometimes return the value already
    converted to dollars; multiplying again would inflate it by 1e6. Guard: if the
    value is already large relative to the scale unit (>= 1000x the unit, far
    beyond any real as-printed line item), assume it is already in dollars and
    leave it untouched.
    """
    factor = scale_factor(detected_scale)
    if factor <= 1.0:
        return value
    return value if abs(value) >= factor * 1000 else value * factor


# --- Segment / region name normalization (used by both extractor and evaluator) -
# Maps common label variants to a canonical key so predictions and ground truth
# compare cleanly. Added to in cycle 2 as part of error analysis.

NAME_ALIASES: Dict[str, str] = {
    "u.s.": "united states",
    "us": "united states",
    "u.s": "united states",
    "domestic": "united states",
    "international": "international",
    "rest of world": "international",
    "row": "international",
    "americas": "americas",
    "emea": "emea",
    "europe, middle east and africa": "emea",
    "apac": "asia pacific",
    "asia-pacific": "asia pacific",
    "greater china": "china",
}


def normalize_name(name: str) -> str:
    """Canonicalize a segment/region key for matching."""
    cleaned = name.strip().lower().strip(".:*†‡()[] ")
    # collapse whitespace
    cleaned = " ".join(cleaned.split())
    return NAME_ALIASES.get(cleaned, cleaned)
