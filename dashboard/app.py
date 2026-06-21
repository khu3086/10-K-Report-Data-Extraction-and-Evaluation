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
            })
        o = m["overall"]
        rows.append({
            "cycle": cycle, "field": "OVERALL",
            "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
        })
    return pd.DataFrame(rows)


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

      .block-container { padding-top: 3rem; padding-bottom: 5rem; max-width: 1160px; }
      hr { margin: 2.4rem 0; border: none; border-top: 1px solid #f0f0f0; }

      /* hero */
      .hero-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
          font-size: 3.1rem; line-height: 1.04; letter-spacing: -0.035em;
          color: #0a0a0a; margin: 0; }
      .hero-sub { color: #9aa0a6; font-size: 1.05rem; line-height: 1.5;
          font-weight: 400; margin-top: 0.4rem; }

      /* section eyebrow + heading */
      .eyebrow { text-transform: uppercase; letter-spacing: 0.12em;
          font-size: 0.72rem; font-weight: 600; color: #6366f1; margin-bottom: 0.35rem; }
      .sec-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600;
          font-size: 1.7rem; letter-spacing: -0.025em; color: #0a0a0a; margin: 0; }
      .sec-desc { color: #9aa0a6; font-size: 0.95rem; margin-top: 0.25rem; }

      /* metric cards */
      div[data-testid="stMetric"] {
          background: #ffffff; border: 1px solid #ededed; border-radius: 18px;
          padding: 22px 24px; transition: box-shadow .2s ease, transform .2s ease;
      }
      div[data-testid="stMetric"]:hover {
          box-shadow: 0 10px 30px rgba(10,10,10,0.05); transform: translateY(-2px); }
      div[data-testid="stMetricLabel"] p {
          font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.09em;
          color: #9aa0a6; font-weight: 600; }
      div[data-testid="stMetricValue"] {
          font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #0a0a0a; }

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
               "scored against a hand-built ground truth.")
    st.divider()
    cycle = st.selectbox(
        "Refinement cycle", cycles, index=len(cycles) - 1,
        help="Each cycle reflects a round of prompt/extraction refinement.",
    )
    st.caption(f"Showing cycle **{cycle}** · {len(cycles)} cycle(s) available.")

df = pd.read_csv(os.path.join(OUT, f"extractions_cycle{cycle}.csv"))
metrics = _load_metrics()

# --- Hero ---------------------------------------------------------------------
h_left, h_right = st.columns([1.5, 1], gap="large")
with h_left:
    st.markdown(
        '<div class="hero-title">Financial data,<br>extracted from 10-Ks.</div>',
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        '<div class="hero-sub">Segment revenue, geographic revenue, and R&D '
        f'expense pulled from filings across {df["ticker"].nunique()} companies — '
        'scored against a hand-built ground truth, cycle over cycle.</div>',
        unsafe_allow_html=True,
    )

st.write("")
st.write("")

# KPI row
overall_now = metrics[(metrics["cycle"] == cycle) & (metrics["field"] == "OVERALL")]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Companies", df["ticker"].nunique())
k2.metric("Fields tracked", df["field"].nunique())
k3.metric("Extracted rows", f"{len(df):,}")
if not overall_now.empty:
    row = overall_now.iloc[0]
    prev = metrics[(metrics["cycle"] == cycle - 1) & (metrics["field"] == "OVERALL")]
    delta = None
    if not prev.empty:
        delta = f"{(row['f1'] - prev.iloc[0]['f1']):+.1%}"
    k4.metric("Overall F1", f"{row['f1']:.1%}", delta=delta)
else:
    k4.metric("Overall F1", "—", help="Run src/evaluate.py to compute metrics.")

st.divider()

# --- Section 1: extracted values across companies -----------------------------
section_header("Extractions", "Fields across companies",
               "Compare totals across the portfolio, then drill into one company.")
field = st.selectbox("Field", sorted(df["field"].unique()), format_func=_label)
sub = df[df["field"] == field].copy()
sub["value_b"] = sub["value"] / 1e9  # billions, for readability

has_breakdown = field != "rd_expense"

if has_breakdown:
    # Each company reports its own region/segment names, so a single multi-series
    # chart is unreadable. Show a comparable total per company, then drill in.
    totals = (sub.groupby(["ticker", "company"], as_index=False)["value_b"].sum()
              .sort_values("value_b", ascending=False))
    overview_title = f"Total {_label(field).lower()} by company ($B)"
else:
    totals = sub.sort_values("value_b", ascending=False)
    overview_title = "R&D expense by company ($B)"

st.caption(overview_title)
fig = px.bar(totals, x="ticker", y="value_b", color_discrete_sequence=[ACCENT])
fig.update_layout(yaxis_title="$ billions", xaxis_title="")
fig.update_traces(hovertemplate="%{x}: $%{y:.1f}B<extra></extra>")
_style_fig(fig)
st.plotly_chart(fig, use_container_width=True)

if has_breakdown:
    tickers = sorted(sub["ticker"].unique())
    pick = st.selectbox("Break down a company", tickers, key="breakdown_ticker")
    comp = (sub[sub["ticker"] == pick]
            .sort_values("value_b", ascending=True))
    st.caption(f"{pick} — {_label(field).lower()} breakdown ($B)")
    bfig = px.bar(
        comp, x="value_b", y="key", orientation="h",
        color_discrete_sequence=[ACCENT],
    )
    bfig.update_layout(xaxis_title="$ billions", yaxis_title="")
    bfig.update_traces(hovertemplate="%{y}: $%{x:.1f}B<extra></extra>")
    _style_fig(bfig, height=max(260, 60 + 34 * len(comp)))
    st.plotly_chart(bfig, use_container_width=True)

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
    metric_name = st.radio(
        "Metric", ["f1", "precision", "recall"],
        horizontal=True, format_func=str.upper,
    )
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
        c1, c2 = st.columns([1, 2])
        with c1:
            cat = errs["category"].value_counts().reset_index()
            cat.columns = ["category", "count"]
            st.caption("Error categories")
            cfig = px.bar(
                cat.sort_values("count"), x="count", y="category",
                orientation="h",
                color_discrete_sequence=[ACCENT],
            )
            cfig.update_layout(xaxis_title="", yaxis_title="")
            _style_fig(cfig, height=340)
            st.plotly_chart(cfig, use_container_width=True)
        with c2:
            st.dataframe(errs, use_container_width=True, height=360, hide_index=True)
    else:
        st.success("No errors recorded for this cycle. 🎉")
else:
    st.info(f"No error log for cycle {cycle}. "
            f"Run `python src/evaluate.py --cycle {cycle}`.")

st.caption("Built for the 10-K extraction OA · data refreshed from `output/`.")
