"""
01_Live_Demo.py
---------------
Live prediction page — enter a news article and get predictions
from LinearSVC, BiLSTM, and the Full Hybrid model.
"""

import os
import sys

import streamlit as st

APP_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(APP_DIR)
for p in [PROJECT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import (  # noqa: E402
    load_classical_models,
    load_bilstm,
    load_hybrid,
    predict_svm,
    predict_bilstm,
    predict_hybrid,
    get_top_tfidf_features,
    extract_linguistic_features,
    clean_text,
    FEATURE_NAMES,
    PALETTE,
)

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Live Demo", page_icon="🔍", layout="wide")
# ---------------------------------------------------------------------------

st.title("Live Fake News Detection Demo")
st.markdown(
    "Paste any news article (or headline + body) below and click **Analyze** "
    "to get predictions from three models."
)

# ── Sample articles ────────────────────────────────────────────────────────
SAMPLE_FAKE = (
    "BREAKING: Secret documents LEAKED reveal massive government cover-up!! "
    "WAKE UP AMERICA — the deep state is hiding the truth about 5G towers and "
    "their connection to the COVID-19 pandemic. Share before they DELETE this! "
    "Anonymous sources inside the Pentagon confirmed everything. YOU WON'T BELIEVE "
    "what they're hiding from you. The mainstream media REFUSES to report this "
    "bombshell revelation that will SHOCK the nation!!!"
)

SAMPLE_REAL = (
    "The Federal Reserve held its benchmark interest rate steady on Wednesday, "
    "as officials said they needed more data before gaining confidence that "
    "inflation is heading sustainably toward their 2% target. The decision to "
    "maintain the federal funds rate in its current target range was unanimous. "
    "Fed Chair Jerome Powell said in a press conference that the central bank "
    "remains attentive to the risks on both sides of its dual mandate."
)

col_sample1, col_sample2, col_sample3 = st.columns([1, 1, 2])
with col_sample1:
    if st.button("Load sample FAKE article", type="secondary"):
        st.session_state["demo_text"] = SAMPLE_FAKE
with col_sample2:
    if st.button("Load sample REAL article", type="secondary"):
        st.session_state["demo_text"] = SAMPLE_REAL

# ── Text input ─────────────────────────────────────────────────────────────
user_text = st.text_area(
    "Article text",
    value=st.session_state.get("demo_text", ""),
    height=200,
    placeholder="Paste a news article here…",
    key="demo_text_area",
)

run = st.button("Analyze", type="primary", disabled=not bool(user_text and user_text.strip()))

if run and user_text and user_text.strip():
    # ── Load models ────────────────────────────────────────────────────────
    with st.spinner("Loading models (first run only)…"):
        tfidf_vec, svm_model          = load_classical_models()
        word_to_idx, bilstm_model     = load_bilstm()
        w2i, h_tfidf, h_scaler, h_model = load_hybrid()

    # ── Run inference ──────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        svm_label,    svm_conf,    cleaned = predict_svm(
            user_text, tfidf_vec, svm_model)
        bilstm_label, bilstm_conf, _       = predict_bilstm(
            user_text, word_to_idx, bilstm_model)
        hybrid_label, hybrid_conf, _       = predict_hybrid(
            user_text, w2i, h_tfidf, h_scaler, h_model)

    # ── Prediction cards ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Predictions")

    def verdict_color(label):
        return PALETTE["Fake"] if label == "Fake" else PALETTE["Real"]

    col1, col2, col3 = st.columns(3)

    with col1:
        color = verdict_color(svm_label)
        st.markdown(
            f"""
            <div style="background:{color}22;border:2px solid {color};
                        border-radius:12px;padding:20px;text-align:center;">
              <h3 style="color:{color};margin:0">LinearSVC</h3>
              <p style="font-size:2rem;font-weight:700;margin:8px 0;color:{color}">
                {svm_label}
              </p>
              <p style="margin:0;color:#555">Confidence: {svm_conf:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        color = verdict_color(bilstm_label)
        st.markdown(
            f"""
            <div style="background:{color}22;border:2px solid {color};
                        border-radius:12px;padding:20px;text-align:center;">
              <h3 style="color:{color};margin:0">BiLSTM</h3>
              <p style="font-size:2rem;font-weight:700;margin:8px 0;color:{color}">
                {bilstm_label}
              </p>
              <p style="margin:0;color:#555">Confidence: {bilstm_conf:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        color = verdict_color(hybrid_label)
        st.markdown(
            f"""
            <div style="background:{color}22;border:2px solid {color};
                        border-radius:12px;padding:20px;text-align:center;">
              <h3 style="color:{color};margin:0">Hybrid (Full)</h3>
              <p style="font-size:2rem;font-weight:700;margin:8px 0;color:{color}">
                {hybrid_label}
              </p>
              <p style="margin:0;color:#555">Confidence: {hybrid_conf:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Agreement check
    all_labels = [svm_label, bilstm_label, hybrid_label]
    if len(set(all_labels)) == 1:
        st.success(f"All three models agree: **{hybrid_label}**")
    else:
        st.warning(
            f"Models disagree — SVM: {svm_label} | BiLSTM: {bilstm_label} | "
            f"Hybrid: {hybrid_label}"
        )

    # ── TF-IDF top features ────────────────────────────────────────────────
    st.divider()
    st.subheader("Top TF-IDF Signal Words")
    st.caption("Highest-weighted terms from the hybrid TF-IDF vectorizer for this article.")

    try:
        import pandas as pd
        import plotly.express as px

        top_feats = get_top_tfidf_features(user_text, h_tfidf, n=20)
        if top_feats:
            words, weights = zip(*top_feats)
            df_feats = pd.DataFrame({"word": words, "weight": weights})
            fig = px.bar(
                df_feats.head(15),
                x="weight",
                y="word",
                orientation="h",
                color_discrete_sequence=[PALETTE["neutral"]],
                labels={"weight": "TF-IDF weight", "word": ""},
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(l=10, r=10, t=30, b=30),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant TF-IDF features found (text may be too short).")
    except ImportError:
        st.info("Install plotly for interactive charts: pip install plotly")

    # ── Linguistic features ────────────────────────────────────────────────
    st.divider()
    st.subheader("Linguistic Feature Profile")
    st.caption("10 handcrafted features — scaled using training-set statistics.")

    try:
        import pandas as pd
        import numpy as np

        ling_raw = extract_linguistic_features(clean_text(user_text), raw_text=user_text)
        df_ling  = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Raw value": [f"{v:.4f}" for v in ling_raw],
        })

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.dataframe(df_ling, use_container_width=True, hide_index=True)

        with col_b:
            import plotly.graph_objects as go

            # Normalize to [0,1] range for radar chart display
            vals = ling_raw.copy().astype(float)
            abs_max = np.abs(vals).max()
            if abs_max > 0:
                vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            else:
                vals_norm = vals

            fig_radar = go.Figure(go.Scatterpolar(
                r=list(vals_norm) + [vals_norm[0]],
                theta=FEATURE_NAMES + [FEATURE_NAMES[0]],
                fill="toself",
                line_color=PALETTE["neutral"],
                fillcolor=PALETTE["neutral"] + "55",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=380,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    except ImportError:
        st.info("Install textblob and textstat for linguistic features.")

    # ── Cleaned text preview ───────────────────────────────────────────────
    with st.expander("View cleaned text"):
        st.text(cleaned)
