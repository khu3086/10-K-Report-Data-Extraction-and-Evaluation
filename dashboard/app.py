"""Streamlit dashboard for the 10-K extraction system.

Three views:
  A. Extracted-field comparison across all companies (current cycle).
  B. Accuracy metrics across refinement cycles (cycle 1 -> cycle 2).
  C. Per-company / per-field error table for live discussion.

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

st.set_page_config(page_title="10-K Extraction Dashboard", layout="wide")
st.title("10-K Report Data Extraction & Evaluation")


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


cycles = _available_cycles()
if not cycles:
    st.warning("No extractions found in `output/`. Run `python src/extract.py --cycle 1` first.")
    st.stop()

cycle = st.sidebar.selectbox("Cycle to display", cycles, index=len(cycles) - 1)
df = pd.read_csv(os.path.join(OUT, f"extractions_cycle{cycle}.csv"))

# --- View A: extracted values across companies --------------------------------
st.header("A. Extracted fields across companies")
field = st.selectbox("Field", sorted(df["field"].unique()))
sub = df[df["field"] == field].copy()
sub["value_b"] = sub["value"] / 1e9  # billions, for readability

if field == "rd_expense":
    fig = px.bar(sub, x="ticker", y="value_b", title="R&D expense ($B)")
else:
    fig = px.bar(
        sub, x="ticker", y="value_b", color="key",
        title=f"{field} by segment/region ($B)", barmode="group",
    )
fig.update_layout(yaxis_title="$ billions", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)
with st.expander("Raw extracted rows"):
    st.dataframe(sub[["company", "ticker", "key", "value", "unit", "source_page"]],
                 use_container_width=True)

# --- View B: accuracy across cycles -------------------------------------------
st.header("B. Accuracy across refinement cycles")
metrics = _load_metrics()
if metrics.empty:
    st.info("No metrics yet. Run `python src/evaluate.py --cycle N` after building ground truth.")
else:
    metric_name = st.radio("Metric", ["f1", "precision", "recall"], horizontal=True)
    fig2 = px.bar(
        metrics, x="field", y=metric_name, color="cycle",
        barmode="group", title=f"{metric_name.upper()} by field, per cycle",
        color_continuous_scale=None,
    )
    fig2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(metrics.pivot_table(index="field", columns="cycle", values=metric_name),
                 use_container_width=True)

# --- View C: error table ------------------------------------------------------
st.header("C. Error analysis")
err_path = os.path.join(OUT, f"errors_cycle{cycle}.csv")
if os.path.exists(err_path):
    errs = pd.read_csv(err_path)
    if not errs.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            cat = errs["category"].value_counts().reset_index()
            cat.columns = ["category", "count"]
            st.plotly_chart(
                px.bar(cat, x="count", y="category", orientation="h",
                       title="Error categories"),
                use_container_width=True,
            )
        with c2:
            st.dataframe(errs, use_container_width=True, height=360)
    else:
        st.success("No errors recorded for this cycle.")
else:
    st.info(f"No error log for cycle {cycle}. Run `python src/evaluate.py --cycle {cycle}`.")
