# 10-K Report Data Extraction & Evaluation

A **hybrid** (rule-based + LLM) system that extracts three challenging numerical
fields from public-company 10-K reports (PDF) and evaluates extraction accuracy
against a hand-built ground-truth dataset — with a measurable iteration story and
a Streamlit dashboard.

## Extracted Fields

Three fields chosen to require interpretation to locate consistently across
filings (so the first pass has real errors to analyze):

| Field (`key`) | Type | Where it lives | Why it's challenging |
|---|---|---|---|
| **Segment revenue** (`segment_revenue`) | multi-value | Segment note | Segment names/count/layout differ per company |
| **Geographic revenue** (`geographic_revenue`) | multi-value | Revenue/segment note | Inconsistent groupings; sometimes in MD&A |
| **R&D expense** (`rd_expense`) | scalar | Income statement / note | Different labels; current-vs-prior-year column confusion |

## How it works

```
EDGAR 10-K (HTML) ──render──▶ PDF ──parse──▶ per-page text + tables
                                                │
                              anchor-keyword scoring  (locate.py)
                                                │
                                  best region per field
                                                │
                         Claude structured extraction  (llm.py, messages.parse)
                                                │
                              normalized values  →  extractions_cycle{N}.csv
                                                │
                          compare vs ground_truth.csv  (evaluate.py)
                                                │
                       metrics_cycle{N}.json + errors_cycle{N}.csv  →  dashboard
```

The **rule-based** step (`locate.py`) scores each page by field anchor keywords and
hands only the best region to the LLM — keeping prompts small and focused. The
**LLM** step (`llm.py`) uses Claude with a forced JSON schema (`messages.parse`),
returning validated values in actual dollars. Default model is `claude-opus-4-8`
(override with `EXTRACT_MODEL`).

## Repository layout

```
companies.yaml            # 10 companies (ticker, CIK, fiscal year)
data/
  reports/                # rendered 10-K PDFs (gitignored; regenerate via edgar_fetch)
  ground_truth.csv        # manually verified values (template provided)
src/
  edgar_fetch.py          # CIK -> latest 10-K -> download HTML -> render PDF (Playwright)
  parse.py                # PDF -> per-page text (PyMuPDF) + tables (pdfplumber)
  locate.py               # anchor scoring -> best region per field
  fields.py               # field defs, anchors, normalization, per-cycle hints  ← live-edit
  llm.py                  # Anthropic client + structured-output schema           ← live-edit
  extract.py              # pipeline CLI -> extractions_cycle{N}.csv
  evaluate.py             # metrics + error log vs ground truth
output/                   # extractions_cycle{N}.csv, metrics_cycle{N}.json, errors_cycle{N}.csv
dashboard/app.py          # Streamlit: field comparison + cycle-over-cycle accuracy + errors
```

## Setup

```bash
pip install -r requirements.txt
playwright install chromium          # for HTML -> PDF rendering
export ANTHROPIC_API_KEY=sk-ant-...  # required for extraction
```

> **Why render to PDF?** EDGAR serves 10-Ks as HTML, not PDF. To satisfy the
> "PDF input" requirement, `edgar_fetch.py` downloads the primary HTML document
> and renders it to PDF with headless Chromium.

## Run

```bash
# 1. Fetch + render 10-Ks  (data/reports/<TICKER>.pdf)
python src/edgar_fetch.py                 # or: --only AAPL,MSFT

# 2. Cycle 1 — baseline extraction (expected to have errors)
python src/extract.py --cycle 1

# 3. Build data/ground_truth.csv by hand (see the template + schema in that file),
#    then evaluate:
python src/evaluate.py --cycle 1          # -> output/metrics_cycle1.json, errors_cycle1.csv

# 4. Cycle 2 — refined extraction, then re-evaluate
python src/extract.py --cycle 2
python src/evaluate.py --cycle 2

# 5. Dashboard
streamlit run dashboard/app.py
```

## Evaluation & metrics

`data/ground_truth.csv` holds the verified value for each `(ticker, field, key)`,
in actual dollars. Comparison uses a 0.5% relative tolerance (absorbs rounding /
unit jitter).

- **Scalar (`rd_expense`)** — accuracy (within tolerance) + value MAPE.
- **Multi-value (segment / geographic)** — precision / recall / F1 over
  `(normalized-key, value-within-tolerance)` pairs, plus value MAPE on matches.

`evaluate.py` also classifies every miss (`scale_x1000`, `wrong_year_or_rounding`,
`missing_key`, `spurious_key`, …) into `errors_cycle{N}.csv` and a category
histogram in the metrics JSON — this drives the error analysis.

## Iteration story

The improvement is encoded in `src/fields.py → CYCLE_HINTS`:

- **Cycle 1 (baseline):** minimal prompt — "report numbers as they appear."
  Produces the expected error classes: wrong scale, prior-year column picked,
  totals counted as segments, segment-name mismatches.
- **Error analysis:** `errors_cycle1.csv` + the category histogram surface the
  dominant failure modes.
- **Cycle 2 (refined):** targeted instructions — detect table scale, force the
  current fiscal-year column, exclude totals/reconciliation rows, normalize
  segment/region names. Re-run and compare `metrics_cycle1` vs `metrics_cycle2`.

Because cycle logic is isolated in `fields.py`/`llm.py`, the system is built for
**live modification** during the presentation — adjust an anchor, a hint, or a
normalization rule and re-run a single field.

## Notes

- Individual work; AI tools used for research and coding support.
- `ANTHROPIC_API_KEY` is required at extraction time; EDGAR requests use a
  descriptive `User-Agent` (override with `SEC_USER_AGENT`).
