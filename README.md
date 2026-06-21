# 10-K Report Data Extraction & Evaluation

A **hybrid** (rule-based + LLM) system that extracts three challenging numerical
fields from public-company 10-K reports (PDF), evaluates them against an
**automatically-built, independent** ground truth, and demonstrates a clean
cycle-1 → cycle-2 improvement. Includes a Streamlit dashboard.

## Live dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=khu3086/10-K-Report-Data-Extraction-and-Evaluation&branch=main&mainModule=dashboard/app.py)

One-click deploy on [Streamlit Community Cloud](https://share.streamlit.io): sign
in with GitHub, point it at this repo (`main` branch, main file
`dashboard/app.py`), and Deploy. The dashboard reads the committed `output/`
metrics, so no API key is needed to view it. Resulting URL looks like
`https://<your-subdomain>.streamlit.app`.

## Results

Ground truth is built automatically (no manual entry): **R&D from SEC XBRL**
(authoritative), **segment/geographic from an independent model** (mistral-small,
different from the gpt-4o-mini system under test). Matching uses a 0.5% value
tolerance with value-aware key matching.

| Field | Cycle 1 F1 | Cycle 2 F1 |
|---|---|---|
| Segment revenue | 0.00 | **0.31** |
| Geographic revenue | 0.00 | **0.45** |
| R&D expense | 0.00 | **0.47** |
| **Overall** | **0.00** | **0.37** |

The cycle-1 baseline reads the printed figures faithfully but **does not
normalize the table's stated scale** — so every value is off by ~10⁶ (the
dominant error category is `scale_x1e6`). Cycle 2 fixes this and lifts overall
F1 from 0 to 0.37. (Exact numbers depend on the LLM run; re-running reproduces
the same direction and magnitude.)

## Extracted fields

| Field (`key`) | Type | Where it lives | Why it's challenging |
|---|---|---|---|
| **Segment revenue** (`segment_revenue`) | multi-value | Segment note | Segment names/count/layout differ per company |
| **Geographic revenue** (`geographic_revenue`) | multi-value | Revenue/segment note | Inconsistent groupings; sometimes in MD&A |
| **R&D expense** (`rd_expense`) | scalar | Income statement / note | Different labels; sub-line vs. headline tagging |

## How it works

```
EDGAR 10-K (HTML) ──requests──▶ render to PDF (Playwright)
        │
        ├─ parse.py     PDF → per-page text (PyMuPDF) + tables (pdfplumber)
        ├─ locate.py    anchor-keyword scoring → best region per field
        ├─ llm.py       ONE structured extraction per field (OpenRouter), number-as-printed + scale
        └─ extract.py   derive BOTH cycles from that one extraction via post-processing
                          ▼
        evaluate.py  vs  data/ground_truth.csv  →  metrics_cycle{1,2}.json + errors_cycle{1,2}.csv
                          ▼
        dashboard/app.py  (Streamlit)
```

**Key design choice — extract once, derive cycles by post-processing.** Each
field is extracted from each filing exactly once with a single stable prompt
(number *as printed* + the table's scale). The two refinement cycles are then
pure deterministic post-processing, so the *only* difference between them is our
code — never the LLM's output. This isolates the iteration (cycle 2 = cycle 1 +
fixes) and halves API cost.

- **Cycle 1 (baseline):** values used as printed (no unit scaling); totals kept.
- **Cycle 2 (refined):** deterministic unit scaling to actual dollars
  (scale detected from the table text, not trusted to the LLM) + totals/
  reconciliation rows excluded + segment/region names normalized.

## Repository layout

```
companies.yaml            # 10 companies (ticker, CIK)
data/
  reports/                # rendered 10-K PDFs (gitignored; regenerate via edgar_fetch)
  ground_truth.csv        # AUTO-built: XBRL (R&D) + independent model (segment/geo)
src/
  edgar_fetch.py          # CIK → latest 10-K → download HTML (requests) → render PDF (Playwright)
  parse.py                # PDF → per-page text + tables
  locate.py               # anchor scoring → best region per field
  fields.py               # field defs, anchors, scale detection/normalization, post-proc rules  ← live-edit
  llm.py                  # OpenRouter client + structured-output schema                          ← live-edit
  extract.py              # one extraction → both cycle CSVs
  build_ground_truth.py   # XBRL R&D + independent-model segment/geographic ground truth
  evaluate.py             # value-aware precision/recall/F1 + MAPE + error categorization
output/                   # extractions_cycle{N}.csv, metrics_cycle{N}.json, errors_cycle{N}.csv
dashboard/app.py          # Streamlit: field comparison + cycle-over-cycle accuracy + errors
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
export OPENROUTER_API_KEY=sk-or-...        # extraction + GT use OpenRouter (OpenAI-compatible)
```

Optional overrides: `EXTRACT_MODEL` (default `openai/gpt-4o-mini`), `GT_MODEL`
(default `openai/gpt-4o`; this project used `mistralai/mistral-small-24b-instruct-2501`),
`SEC_USER_AGENT`.

> **Why render to PDF?** EDGAR serves 10-Ks as HTML; `edgar_fetch.py` downloads
> the primary document with `requests` (a compliant User-Agent — the headless
> browser gets bot-blocked) and renders that local HTML to PDF with Chromium.

## Run

```bash
# 1. Fetch + render all 10 filings  → data/reports/<TICKER>.pdf
python src/edgar_fetch.py

# 2. Extract (one pass writes BOTH cycle CSVs)
python src/extract.py

# 3. Build independent ground truth (XBRL R&D + independent model seg/geo)
GT_MODEL=mistralai/mistral-small-24b-instruct-2501 python src/build_ground_truth.py

# 4. Evaluate both cycles
python src/evaluate.py --cycle 1
python src/evaluate.py --cycle 2

# 5. Dashboard
streamlit run dashboard/app.py
```

## Evaluation

- **R&D** ground truth comes from SEC's XBRL Company-Concept API (authoritative);
  the builder takes the largest candidate R&D tag to handle sub-line tagging
  (e.g. J&J reports headline R&D under `...ExcludingAcquiredInProcessCost`).
- **Segment/geographic** ground truth comes from an *independent* model, so the
  evaluation isn't circular.
- **Matching:** per company+field, a prediction is a true positive if it matches
  a truth cell by normalized key **or** by value within 0.5% (so a correct number
  with a name variant — "greater china" vs "china" — still counts). Reports
  precision / recall / F1 per field and overall, plus value MAPE on matches.
- **Error analysis:** every miss is categorized (`scale_x1e6`, `scale_x1000`,
  `wrong_year_or_rounding`, `spurious_key`, `missing_key`, `wrong_value`) into
  `errors_cycle{N}.csv` and a histogram in the metrics JSON — this is what drives
  the cycle-1 → cycle-2 refinement.

## Iteration story

1. **Cycle 1 (baseline)** extracts the printed numbers but skips unit
   normalization. Error analysis shows `scale_x1e6` dominates (every value off by
   10⁶) → overall F1 ≈ 0.
2. **Cycle 2 (refined)** detects the table scale deterministically from the page
   text, converts to actual dollars (with a double-scaling guard), excludes
   totals/reconciliation rows, and normalizes segment/region names → scale errors
   eliminated, F1 → ~0.37.
3. **Remaining errors** (residual `scale_x1000`, `wrong_value`, `spurious_key`)
   come mostly from cross-model disagreement on segment granularity — the natural
   next refinement (a "cycle 3").

Because cycle logic is isolated in `fields.py`/`extract.py`, the system is built
for **live modification** during the presentation.

## Notes

- Individual work; AI tools used for research and coding support.
- `OPENROUTER_API_KEY` required at extraction/GT time; EDGAR requests use a
  descriptive User-Agent (`SEC_USER_AGENT`).
