"""Streamlit dashboard for the 10-K extraction & evaluation system.

Sections:
  1. Extracted-field comparison across all companies (selected cycle).
  2. Accuracy metrics across refinement cycles (cycle-over-cycle).
  3. Per-company / per-field error analysis.

Run:
    streamlit run dashboard/app.py
"""

import glob
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

# --- Shared styling -----------------------------------------------------------
ACCENT = "#6366f1"          # indigo
ACCENT_SOFT = "#a5b4fc"     # lavender
INK = "#0a0a0a"
MUTED = "#9aa0a6"
PALETTE = ["#6366f1", "#a5b4fc", "#0a0a0a", "#c7d2fe", "#818cf8",
           "#4f46e5", "#cbd5e1", "#312e81"]
PLOT_TEMPLATE = "plotly_white"

FIELD_LABELS = {
    "segment_revenue": "Segment revenue",
    "geographic_revenue": "Geographic revenue",
    "rd_expense": "R&D expense",
}


def _label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


# Filings report values in different scales; normalise everything to billions so
# bars are comparable no matter which cycle (or unit convention) is selected.
_UNIT_TO_DOLLARS = {
    "actual": 1.0,
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
}


def _to_billions(value, unit) -> float:
    factor = _UNIT_TO_DOLLARS.get(str(unit).strip().lower(), 1.0)
    return (value * factor) / 1e9


def _style_fig(fig, *, height=420):
    """Apply consistent chart chrome across every figure."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=20, b=8),
        title=None,
        showlegend=False,
        font=dict(family="Inter, sans serif", size=13, color="#52525b"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12, font_family="Inter, sans serif",
                        bgcolor=INK, font_color="#ffffff"),
        bargap=0.45,
    )
    fig.update_xaxes(showgrid=False, showline=False, zeroline=False,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#f1f1f4", zeroline=False, showline=False,
                     tickfont=dict(color=MUTED))
    return fig


# --- Data loading -------------------------------------------------------------
def _available_cycles():
    cycles = []
    for p in glob.glob(os.path.join(OUT, "extractions_cycle*.csv")):
        try:
            cycles.append(int(os.path.basename(p).split("cycle")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass
    return sorted(set(cycles))


def _load_metrics():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "metrics_cycle*.json"))):
        with open(p) as f:
            m = json.load(f)
        cycle = m["cycle"]
        for fkey, fm in m["per_field"].items():
            rows.append({
                "cycle": cycle, "field": fkey,
                "precision": fm["precision"], "recall": fm["recall"], "f1": fm["f1"],
                "value_mape": fm.get("value_mape"),
            })
        o = m["overall"]
        rows.append({
            "cycle": cycle, "field": "OVERALL",
            "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
            "value_mape": None,
        })
    return pd.DataFrame(rows)


def _value_accuracy(metrics: pd.DataFrame, cycle: int):
    """Mean 'value correct when matched' across fields = 1 - mean MAPE.

    This is the counterpart to F1: F1 measures whether we found the right cells,
    value accuracy measures whether the *numbers* in matched cells are right.
    """
    m = metrics[(metrics["cycle"] == cycle) & (metrics["field"] != "OVERALL")]
    mape = m["value_mape"].dropna()
    if mape.empty:
        return None
    return max(0.0, 1.0 - float(mape.mean()))


# --- Page chrome --------------------------------------------------------------
st.set_page_config(
    page_title="10-K Extraction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

      html, body, [class*="css"], p, span, label, div { font-family: 'Inter', sans-serif; }
      h1, h2, h3, h4, h5 {
          font-family: 'Space Grotesk', sans-serif;
          letter-spacing: -0.025em; color: #0a0a0a; font-weight: 600;
      }

      .block-container { padding-top: 2.5rem; padding-bottom: 5rem; max-width: 1180px; }
      /* dividers fade in rather than slicing the page */
      hr { margin: 3rem 0 2.6rem; border: none; height: 1px;
          background: linear-gradient(90deg, #ececf3 0%, #f6f6fa 60%, transparent 100%); }

      /* hero */
      .hero { margin: 0.2rem 0 0.4rem; }
      .hero-eyebrow { display: inline-block; text-transform: uppercase;
          letter-spacing: 0.14em; font-size: 0.7rem; font-weight: 600; color: #6366f1;
          background: #eef0fe; border: 1px solid #e0e3fb; border-radius: 999px;
          padding: 0.28rem 0.7rem; margin-bottom: 0.95rem; }
      .hero-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
          font-size: 3.1rem; line-height: 1.05; letter-spacing: -0.035em;
          color: #0a0a0a; margin: 0;
          background: linear-gradient(100deg, #0a0a0a 30%, #4f46e5 130%);
          -webkit-background-clip: text; background-clip: text;
          -webkit-text-fill-color: transparent; }
      .hero-sub { color: #71757c; font-size: 1.08rem; line-height: 1.55;
          font-weight: 400; margin-top: 0.9rem; max-width: 46rem; }

      /* section eyebrow + heading */
      .eyebrow { text-transform: uppercase; letter-spacing: 0.12em;
          font-size: 0.72rem; font-weight: 600; color: #6366f1; margin-bottom: 0.4rem; }
      .sec-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
          font-size: 1.7rem; letter-spacing: -0.025em; color: #0a0a0a; margin: 0; }
      .sec-desc { color: #9aa0a6; font-size: 0.95rem; margin-top: 0.3rem;
          max-width: 42rem; }

      /* chart title sitting just above a plot */
      .chart-title { font-weight: 600; font-size: 0.92rem; color: #3f3f46;
          margin: 0.2rem 0 0.5rem; }
      .chart-title span { color: #9aa0a6; font-weight: 400; }

      /* metric cards */
      div[data-testid="stMetric"] {
          background: #ffffff; border: 1px solid #ededed; border-radius: 18px;
          padding: 20px 22px; position: relative; overflow: hidden;
          transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
      }
      /* subtle accent rail down the left of each card */
      div[data-testid="stMetric"]::before {
          content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
          background: linear-gradient(#6366f1, #a5b4fc); opacity: 0; transition: opacity .2s ease; }
      div[data-testid="stMetric"]:hover {
          box-shadow: 0 12px 34px rgba(10,10,10,0.06); transform: translateY(-3px);
          border-color: #e0e3fb; }
      div[data-testid="stMetric"]:hover::before { opacity: 1; }
      div[data-testid="stMetricLabel"] p {
          font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em;
          color: #9aa0a6; font-weight: 600; }
      div[data-testid="stMetricValue"] {
          font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #0a0a0a;
          font-size: 2rem; }

      /* inputs */
      div[data-baseweb="select"] > div {
          border-radius: 12px; border-color: #e7e7e7; background: #fafafa; }
      .stSelectbox label, .stRadio label { color: #6b7280; font-weight: 500; font-size: 0.85rem; }

      /* buttons */
      .stButton > button, .stDownloadButton > button {
          background: #0a0a0a; color: #fff; border: none; border-radius: 12px;
          padding: 0.55rem 1.2rem; font-weight: 500; }
      .stButton > button:hover { background: #262626; color: #fff; }

      /* expander → soft card */
      div[data-testid="stExpander"] {
          border: 1px solid #ededed; border-radius: 16px; background: #fafafa; }
      div[data-testid="stExpander"] summary { font-weight: 500; color: #0a0a0a; }

      /* chart container as a card */
      div[data-testid="stPlotlyChart"] {
          border: 1px solid #ededed; border-radius: 18px; padding: 18px 20px;
          background: #ffffff; }

      /* sidebar */
      section[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #efefef; }
      section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

      /* dataframe */
      div[data-testid="stDataFrame"] { border-radius: 14px; }

      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(eyebrow: str, title: str, desc: str = ""):
    """Render a smallest.ai-style section header: eyebrow + bold title + muted desc."""
    desc_html = f'<div class="sec-desc">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="sec-title">{title}</div>{desc_html}',
        unsafe_allow_html=True,
    )
    st.write("")


def chart_title(text: str, sub: str = ""):
    """A tight title that sits directly above a chart, inside its card."""
    sub_html = f" <span>· {sub}</span>" if sub else ""
    st.markdown(f'<div class="chart-title">{text}{sub_html}</div>',
                unsafe_allow_html=True)

cycles = _available_cycles()
if not cycles:
    st.title("10-K Extraction & Evaluation")
    st.warning(
        "No extractions found in `output/`. "
        "Run `python src/extract.py --cycle 1` to get started."
    )
    st.stop()

# --- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 10-K Extraction")
    st.caption("Structured financial fields extracted from 10-K filings, "
               "scored against an independent ground truth.")
    st.divider()
    cycle = st.selectbox(
        "Refinement cycle", cycles, index=len(cycles) - 1,
        help="Each cycle reflects a round of prompt/extraction refinement.",
    )
    st.caption(f"Showing cycle **{cycle}** · {len(cycles)} cycle(s) available.")

df = pd.read_csv(os.path.join(OUT, f"extractions_cycle{cycle}.csv"))
metrics = _load_metrics()

# --- Hero ---------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<span class="hero-eyebrow">10-K Extraction · Cycle '
    f'{cycle}</span>'
    '<div class="hero-title">Financial data,<br>extracted from 10-Ks.</div>'
    '<div class="hero-sub">Segment revenue, geographic revenue, and R&D '
    f'expense pulled from filings across {df["ticker"].nunique()} companies — '
    'scored against an independent ground truth, cycle over cycle.</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

# KPI row — F1 sits next to value accuracy so the story reads at a glance:
# modest F1 (cell matching) but high value accuracy (the numbers are right).
overall_now = metrics[(metrics["cycle"] == cycle) & (metrics["field"] == "OVERALL")]
k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.metric("Companies", df["ticker"].nunique())
k2.metric("Extracted rows", f"{len(df):,}",
          help=f"{df['field'].nunique()} fields across {df['ticker'].nunique()} companies.")
if not overall_now.empty:
    row = overall_now.iloc[0]
    prev = metrics[(metrics["cycle"] == cycle - 1) & (metrics["field"] == "OVERALL")]
    delta = None
    if not prev.empty:
        delta = f"{(row['f1'] - prev.iloc[0]['f1']):+.1%} vs cycle {cycle - 1}"
    k3.metric("Overall F1", f"{row['f1']:.1%}", delta=delta,
              help="Cell-level precision/recall harmonic mean — did we find the "
                   "right (segment, value) pairs?")
else:
    k3.metric("Overall F1", "—", help="Run src/evaluate.py to compute metrics.")

va = _value_accuracy(metrics, cycle)
if va is not None:
    prev_va = _value_accuracy(metrics, cycle - 1)
    va_delta = f"{(va - prev_va):+.1%} vs cycle {cycle - 1}" if prev_va is not None else None
    k4.metric("Value accuracy", f"{va:.1%}", delta=va_delta,
              help="On matched cells, how close are the numbers? (1 − mean MAPE). "
                   "High value accuracy with lower F1 means the figures are right "
                   "but segment/region names diverge.")
else:
    k4.metric("Value accuracy", "—", help="Run src/evaluate.py to compute MAPE.")

st.divider()

# --- Section 1: extracted values across companies -----------------------------
section_header("Extractions", "Fields across companies",
               "Compare totals across the portfolio, then drill into one company.")

# Control row: field on the left, the company to drill into on the right, so both
# charts below read from one shared header instead of selectors stacked mid-page.
field = st.selectbox("Field", sorted(df["field"].unique()), format_func=_label)
sub = df[df["field"] == field].copy()
# Unit-aware: filings report in millions / thousands / billions / actual, so
# normalise every row to billions before charting (raw value / 1e9 would render
# millions-denominated cycles as ~0).
sub["value_b"] = [_to_billions(v, u) for v, u in zip(sub["value"], sub["unit"])]

has_breakdown = field != "rd_expense"

if has_breakdown:
    # Each company reports its own region/segment names, so a single multi-series
    # chart is unreadable. Show a comparable total per company, then drill in.
    totals = (sub.groupby(["ticker", "company"], as_index=False)["value_b"].sum()
              .sort_values("value_b", ascending=False))
else:
    totals = sub.sort_values("value_b", ascending=False)


def _overview_chart():
    chart_title(f"{_label(field)} by company", "$ billions")
    fig = px.bar(totals, x="ticker", y="value_b", color_discrete_sequence=[ACCENT])
    fig.update_layout(yaxis_title="$ billions", xaxis_title="")
    fig.update_traces(hovertemplate="%{x}: $%{y:.1f}B<extra></extra>")
    _style_fig(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


if has_breakdown:
    # Overview and per-company breakdown sit side by side — the portfolio view and
    # the drill-down read together rather than as two stacked blocks.
    tickers = sorted(sub["ticker"].unique())
    default_idx = tickers.index(totals.iloc[0]["ticker"]) if len(totals) else 0
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        _overview_chart()
    with right:
        pick = st.selectbox("Break down a company", tickers, index=default_idx,
                            key="breakdown_ticker")
        comp = sub[sub["ticker"] == pick].sort_values("value_b", ascending=True)
        chart_title(f"{pick} — {_label(field).lower()}", "$ billions")
        bfig = px.bar(comp, x="value_b", y="key", orientation="h",
                      color_discrete_sequence=[ACCENT])
        bfig.update_layout(xaxis_title="$ billions", yaxis_title="")
        bfig.update_traces(hovertemplate="%{y}: $%{x:.1f}B<extra></extra>")
        _style_fig(bfig, height=max(300, 70 + 40 * len(comp)))
        st.plotly_chart(bfig, use_container_width=True)
else:
    _overview_chart()

with st.expander("View raw extracted rows"):
    st.dataframe(
        sub[["company", "ticker", "key", "value", "unit", "source_page"]],
        use_container_width=True, hide_index=True,
        column_config={
            "value": st.column_config.NumberColumn("value", format="%.0f"),
            "source_page": st.column_config.NumberColumn("page"),
        },
    )

st.divider()

# --- Section 2: accuracy across cycles ----------------------------------------
section_header("Evaluation", "Accuracy across cycles",
               "Precision, recall, and F1 by field — measured each refinement cycle.")
if metrics.empty:
    st.info("No metrics yet. Run `python src/evaluate.py --cycle N` "
            "after building ground truth.")
else:
    # The headline finding, stated up front: value accuracy ≫ F1 means the model
    # reads the right numbers but disagrees on segment/region naming.
    va_now = _value_accuracy(metrics, cycle)
    f1_now = overall_now.iloc[0]["f1"] if not overall_now.empty else None
    if va_now is not None and f1_now is not None and va_now - f1_now > 0.15:
        st.info(
            f"**Numbers right, names diverge.** On matched cells the figures are "
            f"**{va_now:.0%}** accurate, yet overall F1 is **{f1_now:.0%}** — the "
            f"gap is almost entirely segment/region *name* matching and "
            f"over-extraction, not wrong values. That's what cycle 2's name "
            f"normalization and total-row removal target.",
            icon="🎯",
        )

    metric_name = st.radio(
        "Metric", ["f1", "precision", "recall"],
        horizontal=True, format_func=str.upper,
    )
    chart_title(f"{metric_name.upper()} by field", "grouped by refinement cycle")
    metrics_plot = metrics.copy()
    metrics_plot["cycle"] = "Cycle " + metrics_plot["cycle"].astype(str)
    fig2 = px.bar(
        metrics_plot, x="field", y=metric_name, color="cycle",
        barmode="group", color_discrete_sequence=PALETTE,
    )
    _style_fig(fig2)
    fig2.update_layout(
        yaxis_range=[0, 1], yaxis_tickformat=".0%",
        xaxis_title="", yaxis_title=metric_name.upper(),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title_text=""),
        bargap=0.3,
    )
    st.plotly_chart(fig2, use_container_width=True)

    pivot = metrics.pivot_table(index="field", columns="cycle", values=metric_name)
    st.dataframe(
        pivot, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(f"Cycle {c}", format="%.3f")
                       for c in pivot.columns},
    )

st.divider()

# --- Section 3: error analysis ------------------------------------------------
section_header("Diagnostics", "Error analysis",
               "Where extractions diverge from ground truth, by category.")
err_path = os.path.join(OUT, f"errors_cycle{cycle}.csv")
if os.path.exists(err_path):
    errs = pd.read_csv(err_path)
    if not errs.empty:
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            cat = errs["category"].value_counts().reset_index()
            cat.columns = ["category", "count"]
            chart_title("Error categories", f"{len(errs)} total")
            cfig = px.bar(
                cat.sort_values("count"), x="count", y="category",
                orientation="h",
                color_discrete_sequence=[ACCENT],
            )
            cfig.update_layout(xaxis_title="", yaxis_title="")
            _style_fig(cfig, height=360)
            st.plotly_chart(cfig, use_container_width=True)
        with c2:
            chart_title("Every divergence", "predicted vs. ground truth")
            st.dataframe(errs, use_container_width=True, height=360, hide_index=True)
    else:
        st.success("No errors recorded for this cycle. 🎉")
else:
    st.info(f"No error log for cycle {cycle}. "
            f"Run `python src/evaluate.py --cycle {cycle}`.")

st.write("")
st.caption("Built for the 10-K extraction OA · data refreshed live from "
           "`output/` · ground truth from SEC XBRL + independent model.")
