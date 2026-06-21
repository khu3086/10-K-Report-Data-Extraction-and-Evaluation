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

from dataclasses import dataclass
from typing import Dict, List


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

CYCLE_HINTS: Dict[int, str] = {
    # Cycle 1 (baseline): the LLM is asked to do the unit conversion itself.
    # This is where the interesting errors come from — models are unreliable at
    # the arithmetic (off by 1000x) and at picking the current-year column.
    1: (
        "Extract the requested values from the provided 10-K excerpt.\n"
        "Convert every value to ACTUAL US DOLLARS using the table's stated scale "
        "(e.g. '$34,550' under a header 'in millions' -> 34550000000)."
    ),
    # Cycle 2 (refined): error analysis of cycle 1 showed scale-arithmetic and
    # prior-year-column errors dominate. Fix: the LLM returns the number EXACTLY
    # as printed plus the detected scale, and the pipeline applies the multiplier
    # deterministically in Python (see extract.py). The prompt now only handles
    # selection (current year, exclude totals, exact names).
    2: (
        "Extract the requested values from the provided 10-K excerpt. Rules:\n"
        "1. Return each value EXACTLY as printed in the table — do NOT convert "
        "units (e.g. print '34,550' as 34550). Report the table's scale in "
        "'detected_scale'.\n"
        "2. FISCAL YEAR: tables show multiple years side by side. Use ONLY the "
        "most recent fiscal year (the leftmost / current-year column).\n"
        "3. NAMES: use the exact segment/region label, trimmed of footnote marks.\n"
        "4. TOTALS: never include 'Total', 'Consolidated', 'Corporate/Other', or "
        "elimination/reconciliation rows.\n"
        "5. Omit any value not present in the excerpt rather than guessing."
    ),
}


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
