"""
app.py
------
WELFake Fake News Detection — Streamlit home page.
Run with:  streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

# Make project root importable
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "WELFake Demo",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)
# ---------------------------------------------------------------------------

st.title("WELFake Fake News Detection")
st.markdown(
    "**CMSE 928 · Applied Machine Learning · Michigan State University**  \n"
    "Siddhish Nirgude"
)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Project Overview")
    st.markdown(
        """
        This project develops a **3-branch hybrid neural network** for binary fake news
        classification on the [WELFake dataset](https://arxiv.org/abs/2002.01325)
        (72,134 news articles, 0 = Fake, 1 = Real).

        Three complementary feature representations are fused:

        | Branch | Representation | Output dim |
        |--------|---------------|-----------|
        | 1 | TF-IDF (50,000 features) | 128 |
        | 2 | GloVe 100d + BiLSTM (256 units/dir) | 128 |
        | 3 | 10 handcrafted linguistic features | 32 |

        All three branches are concatenated (288-dim) and passed through a
        fusion layer → sigmoid output.
        """
    )

with col2:
    st.subheader("Navigation")
    st.markdown(
        """
        - **Live Demo** — Enter any news article and get real-time predictions
          from three models.
        - **Model Comparison** — Compare accuracy, F1, and ROC-AUC across all
          models including ablation variants.
        - **Explainability** — Inspect TF-IDF feature weights and linguistic
          feature breakdowns for your input.
        """
    )

st.divider()

# ── Performance summary ────────────────────────────────────────────────────
st.subheader("Model Performance Summary")

RESULTS_PATH = os.path.join(PROJECT_DIR, "outputs", "results", "all_results.csv")
ABLATION_PATH = os.path.join(PROJECT_DIR, "outputs", "results", "ablation_results.csv")

try:
    df_all = pd.read_csv(RESULTS_PATH)
    df_abl = pd.read_csv(ABLATION_PATH)

    # Drop the delta column for display
    df_abl_display = df_abl[["Variant", "F1_Macro", "Accuracy", "ROC_AUC"]].copy()
    df_abl_display.columns = ["Model", "F1 Macro", "Accuracy", "ROC-AUC"]
    df_all_display = df_all.rename(columns={
        "Model": "Model", "F1_Macro": "F1 Macro", "ROC_AUC": "ROC-AUC"
    })

    tab1, tab2 = st.tabs(["All Models", "Ablation Study"])

    with tab1:
        st.dataframe(
            df_all_display.style.format(
                {"Accuracy": "{:.4f}", "F1 Macro": "{:.4f}", "ROC-AUC": "{:.4f}"}
            ).background_gradient(subset=["F1 Macro"], cmap="YlOrRd"),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.dataframe(
            df_abl_display.style.format(
                {"Accuracy": "{:.4f}", "F1 Macro": "{:.4f}", "ROC-AUC": "{:.4f}"}
            ).background_gradient(subset=["F1 Macro"], cmap="YlOrRd"),
            use_container_width=True,
            hide_index=True,
        )

except FileNotFoundError:
    st.info("Results CSVs not found. Run notebooks 03–08 first.")

st.divider()

# ── Dataset stats ──────────────────────────────────────────────────────────
st.subheader("Dataset: WELFake")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total articles", "72,134")
c2.metric("Fake (label=0)", "39,793  (55.2%)")
c3.metric("Real (label=1)", "32,341  (44.8%)")
c4.metric("Test set size",  "10,821")
