"""
02_Model_Comparison.py
----------------------
Model comparison dashboard — bar charts and tables for all
trained models including ablation variants.
"""

import os
import sys

import pandas as pd
import streamlit as st

APP_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(APP_DIR)
for p in [PROJECT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import PALETTE  # noqa: E402

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
# ---------------------------------------------------------------------------

st.title("Model Performance Comparison")

RESULTS_DIR   = os.path.join(PROJECT_DIR, "outputs", "results")
FIGURES_DIR   = os.path.join(PROJECT_DIR, "outputs", "figures")

# ── Load CSVs ──────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    all_path = os.path.join(RESULTS_DIR, "all_results.csv")
    abl_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df_all = pd.read_csv(all_path) if os.path.exists(all_path) else None
    df_abl = pd.read_csv(abl_path) if os.path.exists(abl_path) else None
    return df_all, df_abl


df_all, df_abl = load_results()

if df_all is None:
    st.error("outputs/results/all_results.csv not found. Run notebooks 03–08 first.")
    st.stop()

# ── F1 bar chart: all models ───────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("F1 Macro — All Models")

    # Determine order and colors
    model_order = [
        "Random Forest", "XGBoost", "Logistic Regression", "LinearSVC",
        "BiLSTM", "Hybrid",
    ]
    color_map = {m: PALETTE["neutral"] for m in model_order}
    color_map["Hybrid"] = PALETTE["Fake"]  # highlight best model

    df_plot = df_all.copy()
    df_plot["color"] = df_plot["Model"].map(
        lambda m: PALETTE["Fake"] if m == "Hybrid" else PALETTE["neutral"]
    )

    fig_all = px.bar(
        df_plot,
        x="Model",
        y="F1_Macro",
        color="color",
        color_discrete_map="identity",
        text="F1_Macro",
        category_orders={"Model": model_order},
        labels={"F1_Macro": "F1 Macro", "Model": ""},
    )
    fig_all.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_all.update_layout(
        showlegend=False,
        yaxis=dict(range=[0.90, 1.0], title="F1 Macro"),
        height=420,
        plot_bgcolor="white",
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Metric table ───────────────────────────────────────────────────────
    st.subheader("Full Metrics Table")
    df_display = df_all.rename(columns={"F1_Macro": "F1 Macro", "ROC_AUC": "ROC-AUC"})
    st.dataframe(
        df_display.style.format(
            {"Accuracy": "{:.4f}", "F1 Macro": "{:.4f}", "ROC-AUC": "{:.4f}"}
        ).background_gradient(subset=["F1 Macro"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Ablation section ───────────────────────────────────────────────────
    if df_abl is not None:
        st.divider()
        st.subheader("Ablation Study — Branch Contribution")
        st.markdown(
            "Each variant removes one branch of the Full Hybrid to isolate "
            "its contribution. **Delta** = variant F1 minus full hybrid F1."
        )

        abl_order = [
            "No Linguistic (TF-IDF + LSTM)",
            "No LSTM (TF-IDF + Linguistic)",
            "No TF-IDF (LSTM + Linguistic)",
            "Full Hybrid (all 3 branches)",
        ]
        abl_colors = [
            PALETTE["neutral"], PALETTE["neutral"],
            PALETTE["neutral"], PALETTE["Real"],
        ]

        fig_abl = go.Figure()
        for variant, color in zip(abl_order, abl_colors):
            row = df_abl[df_abl["Variant"] == variant]
            if row.empty:
                continue
            fig_abl.add_trace(go.Bar(
                name=variant,
                x=[variant],
                y=row["F1_Macro"].values,
                marker_color=color,
                text=[f"{row['F1_Macro'].values[0]:.4f}"],
                textposition="outside",
            ))

        fig_abl.update_layout(
            showlegend=False,
            yaxis=dict(range=[0.95, 1.0], title="F1 Macro"),
            height=400,
            plot_bgcolor="white",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

        # Ablation table with delta
        df_abl_display = df_abl.rename(columns={
            "Variant": "Variant",
            "F1_Macro": "F1 Macro",
            "ROC_AUC": "ROC-AUC",
            "Delta_vs_Full": "Δ vs Full",
        })
        st.dataframe(
            df_abl_display[["Variant", "F1 Macro", "Accuracy", "ROC-AUC", "Δ vs Full"]]
            .style.format({
                "F1 Macro": "{:.4f}",
                "Accuracy": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Δ vs Full": "{:+.4f}",
            }).background_gradient(subset=["F1 Macro"], cmap="YlOrRd"),
            use_container_width=True,
            hide_index=True,
        )

    # ── Saved figures ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Saved Figures from Notebook Analysis")

    results_fig_dir = os.path.join(FIGURES_DIR, "results")
    ablation_fig_dir = os.path.join(FIGURES_DIR, "ablation")

    fig_paths = []
    for fig_dir in [results_fig_dir, ablation_fig_dir]:
        if os.path.isdir(fig_dir):
            for fname in sorted(os.listdir(fig_dir)):
                if fname.endswith(".png"):
                    fig_paths.append(os.path.join(fig_dir, fname))

    if fig_paths:
        n_cols = 2
        rows   = [fig_paths[i:i+n_cols] for i in range(0, len(fig_paths), n_cols)]
        for row_paths in rows:
            cols = st.columns(n_cols)
            for col, path in zip(cols, row_paths):
                with col:
                    st.image(path, caption=os.path.basename(path), use_container_width=True)
    else:
        st.info("No saved figure PNGs found in outputs/figures/.")

except ImportError:
    st.info("Install plotly for interactive charts: pip install plotly")
    # Fallback: just show tables
    st.dataframe(df_all, use_container_width=True)
    if df_abl is not None:
        st.dataframe(df_abl, use_container_width=True)
