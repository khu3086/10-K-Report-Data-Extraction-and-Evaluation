"""Fetch 10-K filings from SEC EDGAR and render them to PDF.

EDGAR serves 10-Ks as HTML, not PDF, so to satisfy the "PDF input" requirement
we download the primary HTML document and render it to PDF with a headless
Chromium (Playwright). Output: data/reports/<TICKER>.pdf

SEC requires a descriptive User-Agent on every request and rate-limits to ~10
requests/second; we stay well under that.

Usage:
    python src/edgar_fetch.py
    python src/edgar_fetch.py --only AAPL,MSFT
"""

import argparse
import os
import time
from typing import Dict, List, Optional

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# SEC asks for a real contact in the UA string. Override via SEC_USER_AGENT.
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "10-K Extraction Project lokesh.nigam@gmail.com"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

RATE_DELAY = 0.3  # seconds between SEC requests


def _load_companies(path: str) -> List[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["companies"]


def _latest_10k(cik: int) -> Optional[Dict[str, str]]:
    """Return {accession, primary_document} for the most recent 10-K, or None."""
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            }
    return None


def _document_url(cik: int, accession: str, doc: str) -> str:
    return ARCHIVE_URL.format(
        cik=cik, acc_nodash=accession.replace("-", ""), doc=doc
    )


def _fetch_html(url: str) -> str:
    """Download a filing's HTML via requests (SEC blocks the headless browser).

    The data.sec.gov JSON API and the www.sec.gov archive both accept a
    compliant User-Agent over plain requests; the headless-Chromium fetch is
    flagged as an 'undeclared automated tool'. So we fetch here and render the
    HTML locally below.
    """
    headers = dict(HEADERS)
    headers["Host"] = "www.sec.gov"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    if "undeclared automated tool" in r.text[:2000].lower():
        raise RuntimeError("SEC blocked the request (User-Agent rejected)")
    return r.text


def _render_html_to_pdf(html: str, out_path: str) -> None:
    """Render filing HTML to PDF with headless Chromium (local content only).

    We use set_content rather than navigating to the live URL: the browser
    renders the already-downloaded HTML, so SEC's bot filter never sees it.
    Relative resource references (logos, CSS) simply fail fast — the financial
    text and tables are inline in the HTML, which is all we need.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="domcontentloaded", timeout=120_000)
        page.pdf(path=out_path, format="Letter", print_background=True)
        browser.close()


def run(companies_path: str, out_dir: str, only: Optional[List[str]]) -> None:
    companies = _load_companies(companies_path)
    if only:
        wanted = {t.upper() for t in only}
        companies = [c for c in companies if c["ticker"].upper() in wanted]

    os.makedirs(out_dir, exist_ok=True)
    for c in companies:
        ticker, cik = c["ticker"], int(c["cik"])
        out_path = os.path.join(out_dir, f"{ticker}.pdf")
        if os.path.exists(out_path):
            print(f"[{ticker}] already have {out_path} — skipping")
            continue
        try:
            print(f"[{ticker}] looking up latest 10-K (CIK {cik}) ...")
            filing = _latest_10k(cik)
            time.sleep(RATE_DELAY)
            if not filing:
                print(f"[{ticker}] no 10-K found — skipping")
                continue
            url = _document_url(cik, filing["accession"], filing["primary_document"])
            print(f"[{ticker}] downloading {url}")
            html = _fetch_html(url)
            time.sleep(RATE_DELAY)
            print(f"[{ticker}] rendering {len(html):,} bytes -> PDF")
            _render_html_to_pdf(html, out_path)
            print(f"[{ticker}] saved -> {out_path}")
        except Exception as e:
            print(f"[{ticker}] FAILED: {e}")
        time.sleep(RATE_DELAY)


def main():
    ap = argparse.ArgumentParser(description="Download + render SEC 10-Ks to PDF.")
    ap.add_argument("--companies", default=os.path.join(ROOT, "companies.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "reports"))
    ap.add_argument("--only", default=None, help="Comma-separated tickers to fetch")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    run(args.companies, args.out, only)


if __name__ == "__main__":
    main()
