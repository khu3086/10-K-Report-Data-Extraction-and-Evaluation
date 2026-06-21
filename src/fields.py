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

from dataclasses import dataclass, field
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
    1: (
        "Extract the requested values from the provided 10-K excerpt. "
        "Report numbers as they appear."
    ),
    2: (
        "Extract the requested values from the provided 10-K excerpt. Follow "
        "these rules precisely:\n"
        "1. SCALE: detect the table's scale from its header (e.g. 'in millions', "
        "'in thousands') and return every value in ACTUAL DOLLARS "
        "(e.g. '$1,234 million' -> 1234000000).\n"
        "2. FISCAL YEAR: 10-K tables show multiple years side by side. Return "
        "ONLY the most recent fiscal year (the leftmost / current-year column). "
        "Never mix a prior-year figure in.\n"
        "3. SEGMENT/REGION NAMES: use the exact label from the filing, trimmed of "
        "footnote markers and trailing punctuation.\n"
        "4. TOTALS: do NOT include 'Total', 'Consolidated', 'Corporate/Other', or "
        "elimination/reconciliation rows as a segment or region entry.\n"
        "5. If a value is genuinely not present in the excerpt, omit it rather "
        "than guessing."
    ),
}


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
