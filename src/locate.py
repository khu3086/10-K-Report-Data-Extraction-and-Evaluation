"""Locate the most relevant region of a 10-K for a given field.

Rule-based half of the hybrid: score each page by how strongly its text matches
a field's anchor keywords, then return a context window (the best page plus a
little neighboring text) for the LLM to extract from. This keeps the LLM prompt
small and focused instead of dumping a 100+ page filing into context.
"""

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


def _score_page(page: Page, anchors: List[str]) -> int:
    """Count anchor occurrences on a page (text + tables), case-insensitive."""
    hay = (page.text + "\n" + page.tables_text).lower()
    score = 0
    for a in anchors:
        score += hay.count(a.lower())
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
