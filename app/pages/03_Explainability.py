"""
03_Explainability.py
--------------------
Feature visualization and explainability page.
Shows TF-IDF signal words, linguistic feature distributions,
and side-by-side comparison of typical fake vs real articles.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(APP_DIR)
for p in [PROJECT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (  # noqa: E402
    FEATURE_NAMES,
    PALETTE,
    clean_text,
    extract_linguistic_features,
    get_top_tfidf_features,
    load_classical_models,
    load_hybrid,
)

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Explainability", page_icon="🔬", layout="wide")
# ---------------------------------------------------------------------------

st.title("Feature Explainability")
st.markdown(
    "Understand *why* the model makes each prediction by inspecting the "
    "TF-IDF term weights and linguistic feature profile of any article."
)

# ── Input ──────────────────────────────────────────────────────────────────
st.subheader("Enter article text")

SAMPLE_FAKE = (
    "BREAKING: Secret documents LEAKED reveal massive government cover-up!! "
    "WAKE UP AMERICA — the deep state is hiding the truth about 5G towers and "
    "their connection to the COVID-19 pandemic. Share before they DELETE this! "
    "Anonymous sources inside the Pentagon confirmed everything."
)

SAMPLE_REAL = (
    "The Federal Reserve held its benchmark interest rate steady on Wednesday, "
    "as officials said they needed more data before gaining confidence that "
    "inflation is heading sustainably toward their 2% target. The decision to "
    "maintain the federal funds rate in its current target range was unanimous."
)

c1, c2, _ = st.columns([1, 1, 2])
with c1:
    if st.button("Load sample FAKE"):
        st.session_state["explain_text"] = SAMPLE_FAKE
with c2:
    if st.button("Load sample REAL"):
        st.session_state["explain_text"] = SAMPLE_REAL

text_input = st.text_area(
    "Article",
    value=st.session_state.get("explain_text", ""),
    height=160,
    placeholder="Paste article text here…",
    key="explain_text_area",
)

analyze = st.button(
    "Explain", type="primary",
    disabled=not bool(text_input and text_input.strip()),
)

if analyze and text_input and text_input.strip():
    cleaned = clean_text(text_input)

    # ── TF-IDF section ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("TF-IDF Signal Words")
    st.caption(
        "Highest-weighted unigrams and bigrams in the hybrid TF-IDF vocabulary "
        "(50,000 features, sublinear TF scaling, n-gram 1–2)."
    )

    try:
        import plotly.express as px

        _, tfidf_vec, _, _ = load_hybrid()
        top_feats = get_top_tfidf_features(text_input, tfidf_vec, n=25)

        if top_feats:
            words, weights = zip(*top_feats)
            df_tfidf = pd.DataFrame({"term": words, "weight": weights})

            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig = px.bar(
                    df_tfidf.head(20),
                    x="weight",
                    y="term",
                    orientation="h",
                    color_discrete_sequence=[PALETTE["neutral"]],
                    labels={"weight": "TF-IDF weight", "term": ""},
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=480,
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.dataframe(
                    df_tfidf.style.format({"weight": "{:.5f}"}),
                    use_container_width=True,
                    height=480,
                )
        else:
            st.info("No vocabulary matches found. Article may be too short or heavily cleaned.")

    except ImportError:
        st.info("Install plotly for interactive charts.")

    # ── Linguistic features ────────────────────────────────────────────────
    st.divider()
    st.subheader("Linguistic Feature Breakdown")
    st.caption(
        "10 handcrafted features capturing readability, style, sentiment, "
        "and negation patterns."
    )

    ling_raw = extract_linguistic_features(cleaned, raw_text=text_input)
    df_ling  = pd.DataFrame({
        "Feature":     FEATURE_NAMES,
        "Value":       [float(v) for v in ling_raw],
        "Description": [
            "Flesch Reading Ease (higher = easier)",
            "Gunning Fog Index (years of education needed)",
            "Total word count",
            "Avg words per sentence",
            "! and ? per character",
            "ALL-CAPS words / total words",
            "Unique words / total words",
            "Sentiment polarity (−1 to +1)",
            "Sentiment subjectivity (0 to 1)",
            "Count of negation words",
        ],
    })

    try:
        import plotly.graph_objects as go

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.dataframe(
                df_ling[["Feature", "Value", "Description"]]
                .style.format({"Value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_b:
            # Normalize for radar display
            vals = np.array([float(v) for v in ling_raw])
            vmin, vmax = vals.min(), vals.max()
            vals_norm = (vals - vmin) / (vmax - vmin + 1e-9)
            labels    = FEATURE_NAMES + [FEATURE_NAMES[0]]
            r_vals    = list(vals_norm) + [vals_norm[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=r_vals,
                theta=labels,
                fill="toself",
                line_color=PALETTE["neutral"],
                fillcolor=PALETTE["neutral"] + "66",
                name="Article",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
                title="Linguistic Profile (normalized)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    except ImportError:
        st.dataframe(df_ling, use_container_width=True, hide_index=True)

    # ── Fake vs Real reference comparison ─────────────────────────────────
    st.divider()
    st.subheader("Reference: Typical Fake vs Real Profiles")
    st.caption("Mean linguistic feature values from the WELFake training set.")

    # Approximate training means from notebook 06
    reference_data = {
        "Feature":               FEATURE_NAMES,
        "Typical FAKE mean": [
            52.4, 11.8, 312.1, 22.6, 0.0018, 0.021, 0.71, 0.082, 0.42, 2.8
        ],
        "Typical REAL mean": [
            48.9, 13.1, 398.4, 28.4, 0.0005, 0.007, 0.68, 0.051, 0.31, 3.4
        ],
        "This article": [f"{v:.4f}" for v in ling_raw],
    }
    df_ref = pd.DataFrame(reference_data)
    st.dataframe(
        df_ref.style.format({
            "Typical FAKE mean": "{:.4f}",
            "Typical REAL mean": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Cleaned text ───────────────────────────────────────────────────────
    with st.expander("View cleaned text fed to models"):
        st.text(cleaned)

# ── How it works section (always visible) ─────────────────────────────────
st.divider()
st.subheader("How the Hybrid Model Works")

with st.expander("Model architecture", expanded=False):
    st.markdown(
        """
        ```
        Input text
            │
            ├──► TF-IDF (50k features)  → Linear(50000→256, ReLU) → Dropout
            │                            → Linear(256→128, ReLU)  → [128-d]
            │
            ├──► Tokenize → Embedding(30000×100)
            │    → BiLSTM(100→256 per dir, bidirectional)
            │    → GlobalMaxPool(512-d) → Linear(512→128, ReLU)  → [128-d]
            │
            └──► 10 linguistic features (scaled)
                 → Linear(10→32, ReLU) → Dropout              → [32-d]

        Concatenate: [128 + 128 + 32] = 288-d
            → Linear(288→128, ReLU) → Dropout
            → Linear(128→1) → Sigmoid
            → 0 = Fake, 1 = Real
        ```
        """
    )

with st.expander("Linguistic features explained", expanded=False):
    st.markdown(
        """
        | Feature | Why it matters |
        |---------|---------------|
        | **Flesch Reading Ease** | Fake news often uses extreme simplicity or complexity |
        | **Gunning Fog** | Professional journalism targets ~12–14 |
        | **Word count** | Real wire-service articles tend to be longer |
        | **Avg sentence length** | Short punchy sentences signal tabloid writing |
        | **Punctuation density** | Excessive `!!!` is a strong fake signal |
        | **Uppercase ratio** | `ALL CAPS` words signal emotional manipulation |
        | **Lexical diversity** | Repetitive vocabulary indicates formulaic writing |
        | **Sentiment polarity** | Real news is closer to neutral (≈ 0.05) |
        | **Sentiment subjectivity** | Real journalism is more objective (≈ 0.3) |
        | **Negation count** | Fake news sometimes negates established facts repeatedly |
        """
    )
