"""PDF -> per-page text and tables.

Two libraries, two jobs:
  - PyMuPDF (fitz): fast, reliable per-page text extraction for anchoring.
  - pdfplumber: table extraction, rendered back to pipe-delimited text so the
    LLM sees row/column structure rather than a flattened word soup.
"""

from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class Page:
    number: int           # 1-indexed page number
    text: str             # plain text (PyMuPDF)
    tables_text: str      # tables rendered as pipe-delimited rows (pdfplumber)


def _render_table(table) -> str:
    """Render a pdfplumber table (list of rows) as pipe-delimited text."""
    lines = []
    for row in table:
        cells = ["" if c is None else " ".join(str(c).split()) for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(path: str) -> List[Page]:
    """Return a list of Page objects with text and rendered tables."""
    pages: List[Page] = []

    # Text via PyMuPDF
    texts: List[str] = []
    with fitz.open(path) as doc:
        for p in doc:
            texts.append(p.get_text("text"))

    # Tables via pdfplumber (page indices align with PyMuPDF order)
    tables_by_page: List[str] = ["" for _ in texts]
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= len(tables_by_page):
                break
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            rendered = [_render_table(t) for t in tables if t]
            tables_by_page[i] = "\n\n".join(rendered)

    for i, text in enumerate(texts):
        pages.append(
            Page(number=i + 1, text=text, tables_text=tables_by_page[i])
        )
    return pages
