"""
features.py
-----------
Feature engineering pipeline for the WELFake fake news detection dataset.
Includes TF-IDF vectorization with systematic parameter selection
experiments and utility functions for saving and loading transformers.

Author: Siddhish Nirgude
Course: CMSE 928 - Applied Machine Learning
"""

import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


def build_tfidf(
    train_texts,
    val_texts,
    test_texts,
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
):
    """
    Fit TF-IDF vectorizer on training texts only and transform all splits.
    Fitting on training data only prevents data leakage into val and test sets.

    Parameters
    ----------
    train_texts : array-like of str
        Training corpus used to fit the vectorizer.
    val_texts : array-like of str
        Validation corpus to transform with the fitted vectorizer.
    test_texts : array-like of str
        Test corpus to transform with the fitted vectorizer.
    max_features : int, optional
        Maximum number of vocabulary features to retain. Default 30000.
    ngram_range : tuple of (int, int), optional
        Lower and upper boundary of n-gram range. Default (1, 2).
    min_df : int or float, optional
        Minimum document frequency for a term to be included. Default 2.
    max_df : float, optional
        Maximum document frequency above which terms are excluded. Default 0.95.
    sublinear_tf : bool, optional
        Apply sublinear TF scaling (1 + log(tf)). Default True.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    X_train : scipy.sparse matrix
        TF-IDF matrix for the training split.
    X_val : scipy.sparse matrix
        TF-IDF matrix for the validation split.
    X_test : scipy.sparse matrix
        TF-IDF matrix for the test split.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
    )

    # Fit only on training data to avoid leaking statistics from val/test
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)

    vocab_size = len(vectorizer.vocabulary_)

    # Estimate memory in MB using the sparse matrix data arrays
    def sparse_mb(matrix):
        return (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024 ** 2

    print(f"Vocabulary size : {vocab_size:,}")
    print(f"X_train shape   : {X_train.shape}  ({sparse_mb(X_train):.1f} MB)")
    print(f"X_val shape     : {X_val.shape}  ({sparse_mb(X_val):.1f} MB)")
    print(f"X_test shape    : {X_test.shape}  ({sparse_mb(X_test):.1f} MB)")

    return vectorizer, X_train, X_val, X_test


def experiment_max_features(
    train_texts,
    train_labels,
    feature_counts=None,
):
    """
    Find optimal max_features by training Logistic Regression at each
    feature count using 3-fold cross validation. Records mean F1 macro,
    std F1, and training time at each setting.

    Parameters
    ----------
    train_texts : array-like of str
        Training corpus.
    train_labels : array-like of int
        Training labels (0 = Fake, 1 = Real).
    feature_counts : list of int, optional
        Feature counts to evaluate. Default [10000, 20000, 30000, 50000].

    Returns
    -------
    pd.DataFrame
        Columns: max_features, mean_f1, std_f1, time_seconds.
    """
    if feature_counts is None:
        feature_counts = [10000, 20000, 30000, 50000]

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []

    for count in feature_counts:
        t0 = time.time()
        vec = TfidfVectorizer(
            max_features=count,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )
        X = vec.fit_transform(train_texts)
        scores = cross_val_score(lr, X, train_labels, cv=cv, scoring="f1_macro", n_jobs=1)
        elapsed = round(time.time() - t0, 2)
        print(f"max_features={count:>6,}  mean_f1={scores.mean():.4f}  std={scores.std():.4f}  time={elapsed}s")
        rows.append({
            "max_features": count,
            "mean_f1":      round(scores.mean(), 4),
            "std_f1":       round(scores.std(), 4),
            "time_seconds": elapsed,
        })

    return pd.DataFrame(rows)


def experiment_ngram_range(
    train_texts,
    train_labels,
    ngram_options=None,
):
    """
    Compare unigrams only vs unigrams and bigrams using 3-fold CV.
    Uses max_features=30000, min_df=2, max_df=0.95.

    Parameters
    ----------
    train_texts : array-like of str
        Training corpus.
    train_labels : array-like of int
        Training labels.
    ngram_options : list of tuple, optional
        N-gram ranges to evaluate. Default [(1, 1), (1, 2)].

    Returns
    -------
    pd.DataFrame
        Columns: ngram_range, vocab_size, mean_f1, std_f1.
    """
    if ngram_options is None:
        ngram_options = [(1, 1), (1, 2)]

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []

    for ngram in ngram_options:
        vec = TfidfVectorizer(
            max_features=30000,
            ngram_range=ngram,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )
        X = vec.fit_transform(train_texts)
        vocab_size = len(vec.vocabulary_)
        scores = cross_val_score(lr, X, train_labels, cv=cv, scoring="f1_macro", n_jobs=1)
        print(f"ngram={ngram}  vocab={vocab_size:,}  mean_f1={scores.mean():.4f}  std={scores.std():.4f}")
        rows.append({
            "ngram_range": str(ngram),
            "vocab_size":  vocab_size,
            "mean_f1":     round(scores.mean(), 4),
            "std_f1":      round(scores.std(), 4),
        })

    return pd.DataFrame(rows)


def experiment_df_params(
    train_texts,
    train_labels,
    df_combinations=None,
):
    """
    Test three min_df and max_df combinations using 3-fold CV.

    Default combinations tested:
      - {min_df: 1, max_df: 0.95}
      - {min_df: 2, max_df: 0.95}
      - {min_df: 5, max_df: 0.90}

    Uses max_features=30000 and ngram_range=(1, 2) for all runs.

    Parameters
    ----------
    train_texts : array-like of str
        Training corpus.
    train_labels : array-like of int
        Training labels.
    df_combinations : list of dict, optional
        Each dict must have keys 'min_df' and 'max_df'.

    Returns
    -------
    pd.DataFrame
        Columns: min_df, max_df, vocab_size, mean_f1, std_f1.
    """
    if df_combinations is None:
        df_combinations = [
            {"min_df": 1, "max_df": 0.95},
            {"min_df": 2, "max_df": 0.95},
            {"min_df": 5, "max_df": 0.90},
        ]

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []

    for combo in df_combinations:
        min_df = combo["min_df"]
        max_df = combo["max_df"]
        vec = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )
        X = vec.fit_transform(train_texts)
        vocab_size = len(vec.vocabulary_)
        scores = cross_val_score(lr, X, train_labels, cv=cv, scoring="f1_macro", n_jobs=1)
        print(
            f"min_df={min_df}  max_df={max_df}  "
            f"vocab={vocab_size:,}  mean_f1={scores.mean():.4f}  std={scores.std():.4f}"
        )
        rows.append({
            "min_df":      min_df,
            "max_df":      max_df,
            "vocab_size":  vocab_size,
            "mean_f1":     round(scores.mean(), 4),
            "std_f1":      round(scores.std(), 4),
        })

    return pd.DataFrame(rows)


def run_gridsearch(X_train, y_train, model, param_grid, model_name, cv=5):
    """
    Run GridSearchCV for a given model and parameter grid.
    Uses stratified k-fold cross validation on training data only.
    Scoring metric is f1_macro to account for class imbalance.

    Parameters
    ----------
    X_train : array-like or sparse matrix
        Training feature matrix.
    y_train : array-like of int
        Training labels.
    model : sklearn estimator
        Unfitted sklearn-compatible estimator.
    param_grid : dict
        Dictionary of hyperparameter names to lists of values to search.
    model_name : str
        Display name for the model, used in printed output.
    cv : int, optional
        Number of stratified cross-validation folds. Default 5.

    Returns
    -------
    best_estimator : sklearn estimator
        Refitted estimator using best parameters on full training data.
    best_params : dict
        Best hyperparameter combination found.
    best_score : float
        Best mean cross-validated f1_macro score.
    cv_results_df : pd.DataFrame
        Full cross-validation results table.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score  = round(grid_search.best_score_, 4)

    print(f"\n{model_name} Grid Search Results")
    print(f"  Best params : {best_params}")
    print(f"  Best CV F1  : {best_score}")

    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return grid_search.best_estimator_, best_params, best_score, cv_results_df


def save_model(model, filepath):
    """
    Save a fitted sklearn model or transformer to disk using joblib.
    Creates parent directories if they do not exist.

    Parameters
    ----------
    model : object
        Any fitted sklearn estimator or transformer (e.g. TfidfVectorizer,
        LogisticRegression).
    filepath : str
        Destination file path including filename (e.g. models/tfidf_lr.joblib).

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)
    size_kb = round(os.path.getsize(filepath) / 1024, 1)
    print(f"Saved: {filepath} ({size_kb} KB)")


def load_model(filepath):
    """
    Load a fitted sklearn model or transformer from disk using joblib.
    Raises FileNotFoundError if the file does not exist.

    Parameters
    ----------
    filepath : str
        Path to the saved joblib file.

    Returns
    -------
    object
        Loaded sklearn estimator or transformer.

    Raises
    ------
    FileNotFoundError
        If no file exists at the given filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at: {filepath}")
    model = joblib.load(filepath)
    print(f"Loaded: {filepath}")
    return model
