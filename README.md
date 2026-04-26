
# Fake News Detection using a Hybrid NLP Model on WELFake Dataset

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-green.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Presentation

Download the final presentation slides:
[Download Presentation (PPTX)](https://github.com/SiddhishNirgude/fake-news-detection-welfake/raw/main/report/presentation.pptx)

## Overview

This project builds a lightweight 3-branch hybrid NLP classification
system for fake news detection on the WELFake dataset. The system
combines three complementary feature representations:

- **Branch 1**: TF-IDF statistical features (50k unigrams + bigrams)
- **Branch 2**: GloVe 100d pretrained embeddings + BiLSTM sequential encoding
- **Branch 3**: 10 handcrafted linguistic writing-style features

**Best Result: 98.62% F1 Macro** — outperforms all published baselines
including Padalko et al. (2024) Att-BiLSTM (97.66%).

All improvements confirmed statistically significant via McNemar's test
(p < 0.0001 for all 4 model pairs).

---

## Results Summary

| Model | F1 Macro | ROC-AUC |
|---|---|---|
| Random Forest | 0.9429 | 0.9879 |
| XGBoost | 0.9627 | 0.9938 |
| Logistic Regression | 0.9703 | 0.9964 |
| LinearSVC | 0.9720 | 0.9967 |
| BiLSTM (learned embeddings) | 0.9783 | 0.9980 |
| Full Hybrid (3 branches) | 0.9843 | 0.9991 |
| **No-Linguistic Hybrid (best)** | **0.9862** | — |

All improvements statistically significant — McNemar p < 0.0001.

---

## Model Architecture

```
Raw Text Article
      │
      ├──────────────────┬──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
Branch 1            Branch 2           Branch 3
TF-IDF (50k)        GloVe (100d)       Linguistic
→ Dense(256)        → BiLSTM(256)      (10 features)
→ Dense(128)        → Dense(128)       → Dense(32)
      │                  │                  │
      └──────────────────┴──────────────────┘
                         │
                  Concatenate (288d)
                         │
                  Dense(128) + Dropout(0.5)
                         │
                  Dense(1, sigmoid)
                         │
                 Fake (0) / Real (1)
```

---

## Dataset

Download WELFake dataset from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Place at: `data/raw/WELFake_Dataset.csv`

The dataset (234 MB) is not included due to GitHub size limits.

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/SiddhishNirgude/fake-news-detection-welfake.git
cd fake-news-detection-welfake
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Place WELFake_Dataset.csv in `data/raw/`

### 4. Run notebooks in order

| Notebook | Description | Platform |
|---|---|---|
| 01_EDA.ipynb | Exploratory data analysis | Mac/Local |
| 02_preprocessing.ipynb | Cleaning and splitting | Mac/Local |
| 03_tfidf_classical.ipynb | TF-IDF + classical ML | HPCC/Colab |
| 04_bilstm_baseline.ipynb | BiLSTM baseline | Colab T4 GPU |
| 05_glove_features.ipynb | GloVe embedding matrix | Colab CPU |
| 06_linguistic_features.ipynb | Linguistic features | Colab CPU |
| 07_hybrid_model.ipynb | 3-branch hybrid model | Colab T4 GPU |
| 08_ablation.ipynb | Ablation study | Colab T4 GPU |
| 09_significance.ipynb | McNemar significance | Colab CPU |

### 5. Run Streamlit app
```bash
streamlit run app/app.py
```

---

## Key Findings

1. **GloVe/BiLSTM is the most critical branch** — removing it drops F1 by 1.65%
2. **Linguistic features hurt slightly** — punctuation and uppercase signals
   lost during preprocessing (key limitation)
3. **Best model**: No-Linguistic Hybrid at 98.62% F1 (TF-IDF + GloVe/BiLSTM)
4. **Beats published SOTA**: Padalko et al. (2024) Att-BiLSTM 97.66%

---

## Graduate Requirements (CMSE 928)

**Ablation Study**: 3 two-branch variants trained to quantify each branch
contribution. GloVe/BiLSTM proved most critical — removing it drops F1 by 1.65%.

**Statistical Significance**: McNemar's test on 4 model pairs. All improvements
significant at alpha=0.05 (p < 0.0001).

---

## Project Structure

```
├── notebooks/           9 Jupyter notebooks (full pipeline)
├── src/
│   ├── preprocess.py    Data cleaning and splitting
│   ├── features.py      Feature extraction
│   ├── models.py        Model architectures
│   ├── evaluate.py      Metrics and evaluation
│   ├── utils.py         Utilities and helpers
│   └── visualize.py     Plotting functions
├── app/                 Streamlit demo application
│   ├── app.py           Home page
│   └── pages/           Live Demo, Comparison, Explainability
├── outputs/
│   ├── figures/         All generated plots
│   └── results/         All result CSVs
├── models/              Small model artifacts
├── report/              Final report and presentation
├── requirements.txt
└── README.md
```

---

## References

- Verma et al. (2021). WELFake. IEEE Trans. Computational Social Systems, 8(4).
- Padalko et al. (2024). WELFake: Enhancing Fake News Detection. Data, 9(1).
- Kausar et al. (2022). Word Embedding with RNN for Fake News Detection.
- Pennington et al. (2014). GloVe: Global Vectors for Word Representation.
- Garg & Sharma (2022). Linguistic Feature-based Learning for Fake News.

---

## Author

**Siddhish Nirgude**
MS Data Science — Michigan State University
CMSE 928 Applied Machine Learning — April 2026

Step 3: Commit and push:
git add README.md
git commit -m "Add full project README with results and setup instructions"
git push origin main

Step 4: Confirm push was successful.
Print: "README updated and pushed to GitHub."
```

---

Come back with confirmation then refresh GitHub to check.

## Large Files Not Included in Repository

The following files are excluded due to GitHub size limits.
Generate them by running the notebooks in order.

### Dataset
| File | Size | Download |
|---|---|---|
| data/raw/WELFake_Dataset.csv | 234 MB | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |

### Processed Data (generated by notebook 02)
| File | Size |
|---|---|
| data/processed/train_clean.csv | 281 MB |
| data/processed/val_clean.csv | 60 MB |
| data/processed/test_clean.csv | 60 MB |
| data/processed/X_train_linguistic.npy | 1.7 MB |
| data/processed/X_val_linguistic.npy | 367 KB |
| data/processed/X_test_linguistic.npy | 367 KB |

### Trained Model Weights (generated by running notebooks)
| File | Size | Generated by |
|---|---|---|
| models/glove_embedding_matrix.npy | 11 MB | notebook 05 |
| models/bilstm_model.pt | 18 MB | notebook 04 |
| models/hybrid_model.pt | 64 MB | notebook 07 |
| models/hybrid_tfidf_vectorizer.joblib | 1.9 MB | notebook 07 |
| models/tfidf_vectorizer.joblib | 1.9 MB | notebook 03 |
| models/model_rf.joblib | 70 MB | notebook 03 |
| models/ablation_no_linguistic.pt | 64 MB | notebook 08 |
| models/ablation_no_lstm.pt | 49 MB | notebook 08 |
| models/ablation_no_tfidf.pt | 15 MB | notebook 08 |

### GloVe Embeddings (downloaded automatically in notebook 05)
| File | Size | Download |
|---|---|---|
| GloVe 6B 100d vectors | 822 MB | [Stanford NLP](https://nlp.stanford.edu/data/glove.6B.zip) |
