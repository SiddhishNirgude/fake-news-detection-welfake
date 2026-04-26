"""
utils.py
--------
Shared model loading, text preprocessing, and inference utilities
for the WELFake Streamlit demo application.

All model loading functions are decorated with @st.cache_resource
so models are loaded once per session and reused across pages.
"""

import os
import re
import sys

import joblib
import numpy as np
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
SRC_DIR     = os.path.join(PROJECT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LEN        = 300
VOCAB_SIZE     = 30000
BILSTM_EMB_DIM = 128
BILSTM_LSTM    = 256
HYBRID_EMB_DIM = 100
HYBRID_LSTM    = 256
HYBRID_TFIDF   = 50000
HYBRID_LING    = 10
DROPOUT        = 0.5

PALETTE = {
    "Fake":    "#E76F51",
    "Real":    "#2A9D8F",
    "neutral": "#E9C46A",
}

FEATURE_NAMES = [
    "flesch_reading_ease",
    "gunning_fog",
    "word_count",
    "avg_sentence_length",
    "punctuation_density",
    "uppercase_ratio",
    "lexical_diversity",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "negation_count",
]

NEGATION_WORDS = {
    "not", "no", "never", "none", "nothing", "neither",
    "nor", "nobody", "nowhere", "without", "hardly",
    "barely", "scarcely", "cannot", "can't", "won't",
    "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't",
}

# ---------------------------------------------------------------------------
# Text cleaning (mirrors src/preprocess.py clean_text)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_pad(text: str, word_to_idx: dict, max_len: int = MAX_LEN) -> np.ndarray:
    tokens = text.split()[:max_len]
    ids = [word_to_idx.get(t, word_to_idx.get("<UNK>", 1)) for t in tokens]
    # Pad with 0 (<PAD>) to max_len
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Linguistic feature extraction
# ---------------------------------------------------------------------------

def extract_linguistic_features(text: str, raw_text: str = None) -> np.ndarray:
    try:
        import textstat
        from textblob import TextBlob
    except ImportError:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    if not isinstance(text, str) or len(text.strip()) < 10:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    flesch = textstat.flesch_reading_ease(text)
    fog    = textstat.gunning_fog(text)

    words   = text.split()
    n_words = len(words)
    if n_words == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    wps = [len(s.split()) for s in sentences if s.strip()]
    avg_sent_len = float(np.mean(wps)) if wps else 0.0

    src = raw_text if raw_text else text
    n_punct       = src.count("!") + src.count("?")
    punct_density = n_punct / max(len(src), 1)
    words_raw       = src.split()
    n_upper         = sum(1 for w in words_raw if w.isupper() and len(w) > 1)
    uppercase_ratio = n_upper / max(len(words_raw), 1)

    unique_words = set(w.lower() for w in words)
    lexical_div  = len(unique_words) / n_words

    blob         = TextBlob(text)
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    words_lower = [w.lower() for w in words]
    neg_count   = sum(1 for w in words_lower if w in NEGATION_WORDS)

    return np.array([
        flesch, fog, n_words, avg_sent_len,
        punct_density, uppercase_ratio, lexical_div,
        polarity, subjectivity, neg_count,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading classical models…")
def load_classical_models():
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    svm = joblib.load(os.path.join(MODELS_DIR, "model_svm.joblib"))
    return vec, svm


@st.cache_resource(show_spinner="Loading BiLSTM model…")
def load_bilstm():
    from models import build_bilstm

    word_to_idx = joblib.load(os.path.join(MODELS_DIR, "tokenizer.joblib"))
    model       = build_bilstm(
        vocab_size    = VOCAB_SIZE,
        embedding_dim = BILSTM_EMB_DIM,
        lstm_units    = BILSTM_LSTM,
        dropout       = DROPOUT,
    )
    state = torch.load(
        os.path.join(MODELS_DIR, "bilstm_model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return word_to_idx, model


@st.cache_resource(show_spinner="Loading Hybrid model…")
def load_hybrid():
    from models import build_hybrid_model

    word_to_idx = joblib.load(os.path.join(MODELS_DIR, "tokenizer.joblib"))
    tfidf_vec   = joblib.load(os.path.join(MODELS_DIR, "hybrid_tfidf_vectorizer.joblib"))
    ling_scaler = joblib.load(os.path.join(MODELS_DIR, "linguistic_scaler.joblib"))

    model = build_hybrid_model(
        tfidf_dim     = HYBRID_TFIDF,
        vocab_size    = VOCAB_SIZE,
        embedding_dim = HYBRID_EMB_DIM,
        lstm_units    = HYBRID_LSTM,
        linguistic_dim= HYBRID_LING,
        dropout       = DROPOUT,
    )
    state = torch.load(
        os.path.join(MODELS_DIR, "hybrid_model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return word_to_idx, tfidf_vec, ling_scaler, model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_svm(raw_text: str, tfidf_vec, svm_model):
    cleaned = clean_text(raw_text)
    vec     = tfidf_vec.transform([cleaned])
    pred    = svm_model.predict(vec)[0]
    score   = svm_model.decision_function(vec)[0]
    # Convert decision boundary distance to a rough [0,1] confidence
    prob    = float(1 / (1 + np.exp(-score)))
    label   = "Real" if pred == 1 else "Fake"
    conf    = prob if pred == 1 else 1 - prob
    return label, conf, cleaned


def predict_bilstm(raw_text: str, word_to_idx: dict, model):
    cleaned  = clean_text(raw_text)
    seq      = tokenize_and_pad(cleaned, word_to_idx)
    x        = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        prob = model(x).squeeze().item()
    label = "Real" if prob >= 0.5 else "Fake"
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, conf, cleaned


def predict_hybrid(raw_text: str, word_to_idx: dict, tfidf_vec, ling_scaler, model):
    cleaned  = clean_text(raw_text)

    # TF-IDF branch
    tfidf_sparse = tfidf_vec.transform([cleaned])
    tfidf_dense  = torch.tensor(
        tfidf_sparse.toarray().astype(np.float32), dtype=torch.float32
    )

    # Sequential branch
    seq  = tokenize_and_pad(cleaned, word_to_idx)
    x_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)

    # Linguistic branch
    ling_raw    = extract_linguistic_features(cleaned, raw_text=raw_text)
    ling_scaled = ling_scaler.transform(ling_raw.reshape(1, -1))
    x_ling      = torch.tensor(ling_scaled.astype(np.float32), dtype=torch.float32)

    with torch.no_grad():
        prob = model(tfidf_dense, x_seq, x_ling).squeeze().item()

    label = "Real" if prob >= 0.5 else "Fake"
    conf  = prob if prob >= 0.5 else 1 - prob
    return label, conf, cleaned


def get_top_tfidf_features(raw_text: str, tfidf_vec, n: int = 15):
    cleaned  = clean_text(raw_text)
    vec      = tfidf_vec.transform([cleaned])
    feature_names = np.array(tfidf_vec.get_feature_names_out())
    row      = vec.toarray().squeeze()
    top_idx  = row.argsort()[::-1][:n]
    return [(feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]
