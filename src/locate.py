"""Locate the most relevant region of a 10-K for a given field.

Rule-based half of the hybrid: score each page by how strongly its text matches
a field's anchor keywords, then return a context window (the best page plus a
little neighboring text) for the LLM to extract from. This keeps the LLM prompt
small and focused instead of dumping a 100+ page filing into context.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from fields import Field
from parse import Page


@dataclass
class Region:
    field_key: str
    page_number: int          # best-matching page (1-indexed)
    score: int                # anchor hit score, for debugging
    context: str              # text + tables passed to the LLM


# A "financial figure": 4+ digit run, optionally comma-grouped (e.g. 178,353).
_FIGURE = re.compile(r"\d[\d,]{3,}")


def _score_page(page: Page, anchors: List[str]) -> int:
    """Score a page by anchor matches, biased toward pages that hold the actual
    numeric table rather than narrative that merely *mentions* the topic.

    Two refinements over a plain keyword count, both motivated by error analysis
    (e.g. Apple's segment table lives in a Notes page that the MD&A narrative was
    out-scoring):
      - anchor hits inside extracted tables count double (that's where data is);
      - a page whose anchors sit alongside a dense figure table gets a bonus,
        so "net sales by reportable segment <table>" beats a prose paragraph.
    """
    text = page.text.lower()
    tables = page.tables_text.lower()
    score = 0
    for a in anchors:
        a = a.lower()
        score += text.count(a)
        score += 2 * tables.count(a)
    if score > 0:
        figures = len(_FIGURE.findall(page.text + " " + page.tables_text))
        if figures >= 8:        # a real financial table, not a passing mention
            score += 3
    return score


def locate(field: Field, pages: List[Page], window: int = 1) -> Optional[Region]:
    """Return the best Region for `field`, or None if no anchor ever matched.

    `window` controls how many neighboring pages of context to include on each
    side of the best page (segment/geographic tables sometimes span a page break).
    """
    scored = [(p, _score_page(p, field.anchors)) for p in pages]
    scored = [(p, s) for (p, s) in scored if s > 0]
    if not scored:
        return None

    best_page, best_score = max(scored, key=lambda ps: ps[1])
    idx = best_page.number - 1

    lo = max(0, idx - window)
    hi = min(len(pages), idx + window + 1)
    chunks = []
    for p in pages[lo:hi]:
        chunks.append(f"--- page {p.number} (text) ---\n{p.text}")
        if p.tables_text:
            chunks.append(f"--- page {p.number} (tables) ---\n{p.tables_text}")
    context = "\n\n".join(chunks)

    return Region(
        field_key=field.key,
        page_number=best_page.number,
        score=best_score,
        context=context,
    )
