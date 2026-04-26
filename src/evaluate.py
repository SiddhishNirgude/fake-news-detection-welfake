"""
evaluate.py
-----------
Evaluation utilities for the WELFake fake news detection project.
Includes functions for computing classification metrics, generating
comparison tables, and running McNemar's statistical significance test.

Author: Siddhish Nirgude
Course: CMSE 928 - Applied Machine Learning
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_score, model_name):
    """
    Compute all classification metrics for a single model.
    Returns a dictionary of metrics for easy aggregation into
    a results table across multiple models.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_score : array-like
        Predicted probability scores or decision function values
        used for ROC-AUC computation.
    model_name : str
        Name of the model for labeling results.

    Returns
    -------
    dict
        Dictionary with keys:
        Model, Accuracy, Precision, Recall, F1_Macro, ROC_AUC.
        All metric values rounded to 4 decimal places.
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ROC-AUC requires continuous scores, not hard predictions
    roc_auc = roc_auc_score(y_true, y_score)

    return {
        "Model":     model_name,
        "Accuracy":  round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1_Macro":  round(f1_macro, 4),
        "ROC_AUC":   round(roc_auc, 4),
    }


def build_results_table(results_list):
    """
    Build a formatted results DataFrame from a list of metric
    dictionaries produced by compute_metrics.
    Sorts by F1_Macro descending so best model appears first.

    Parameters
    ----------
    results_list : list of dict
        Each dict is the output of compute_metrics for one model.

    Returns
    -------
    pd.DataFrame
        Sorted results table with all metrics as columns.
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1_Macro", ascending=False).reset_index(drop=True)

    # Shift index to start at 1 for display readability
    df.index = df.index + 1
    print(df.to_string())
    df.index = range(len(df))

    return df


def save_results(results_df, filepath):
    """
    Save results DataFrame to CSV file.
    Creates parent directories if they do not exist.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table from build_results_table.
    filepath : str
        Full path including filename and .csv extension.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def mcnemar_test(
    y_true,
    y_pred_model_a,
    y_pred_model_b,
    model_a_name,
    model_b_name,
):
    """
    Perform McNemar's test to assess whether two models differ
    significantly in their predictions on the same test set.

    McNemar's test is appropriate here because both models are
    evaluated on the same test examples, making predictions
    dependent rather than independent.

    The null hypothesis is that both models make the same
    proportion of errors. A p-value below 0.05 indicates
    the difference in performance is statistically significant.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred_model_a : array-like
        Predictions from model A.
    y_pred_model_b : array-like
        Predictions from model B.
    model_a_name : str
        Display name for model A.
    model_b_name : str
        Display name for model B.

    Returns
    -------
    dict
        Keys: model_a, model_b, n01, n10, chi2_statistic,
        p_value, significant.
        n01 = model A correct, model B wrong.
        n10 = model A wrong, model B correct.
        significant = True if p_value < 0.05.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_model_a)
    y_pred_b = np.asarray(y_pred_model_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # Discordant cells: one model right, the other wrong
    n01 = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))   # A wrong,  B correct
    n00 = int(np.sum(~correct_a & ~correct_b))  # both wrong
    n11 = int(np.sum(correct_a & correct_b))    # both correct

    # 2x2 contingency table for McNemar's test
    contingency_table = np.array([[n00, n01], [n10, n11]])

    # Yates' continuity correction is applied via correction=True
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table, correction=True)
    significant = p_value < 0.05

    result_label = "Significant difference" if significant else "No significant difference"

    print(f"\nMcNemar Test: {model_a_name} vs {model_b_name}")
    print(f"  n01 (A correct, B wrong) = {n01}")
    print(f"  n10 (A wrong, B correct) = {n10}")
    print(f"  chi2 = {chi2_stat:.4f},  p = {p_value:.6f}")
    print(f"  Result: {result_label} (alpha=0.05)")

    return {
        "model_a":        model_a_name,
        "model_b":        model_b_name,
        "n01":            n01,
        "n10":            n10,
        "chi2_statistic": round(chi2_stat, 4),
        "p_value":        round(p_value, 6),
        "significant":    significant,
    }


def run_all_mcnemar_tests(y_true, predictions_dict):
    """
    Run McNemar's test for all meaningful model pairs and
    return a summary DataFrame.

    Meaningful pairs for this project:
    - Best classical model vs BiLSTM baseline
    - BiLSTM baseline vs Hybrid model
    - Best classical model vs Hybrid model

    Parameters
    ----------
    y_true : array-like
        True labels.
    predictions_dict : dict
        Keys are model names, values are prediction arrays.
        Example: {'LR': y_pred_lr, 'BiLSTM': y_pred_bilstm,
                   'Hybrid': y_pred_hybrid}

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        Model_A, Model_B, chi2, p_value, Significant.
        Sorted by p_value ascending.
    """
    model_names = list(predictions_dict.keys())

    # Canonical model roles used to define meaningful comparison pairs
    # Assumes dict key ordering: classical model first, then BiLSTM, then Hybrid
    best_classical = model_names[0]
    bilstm         = model_names[1] if len(model_names) > 1 else model_names[0]
    hybrid         = model_names[2] if len(model_names) > 2 else model_names[-1]

    pairs = [
        (best_classical, bilstm),
        (bilstm,         hybrid),
        (best_classical, hybrid),
    ]

    rows = []
    for name_a, name_b in pairs:
        result = mcnemar_test(
            y_true,
            predictions_dict[name_a],
            predictions_dict[name_b],
            name_a,
            name_b,
        )
        rows.append({
            "Model_A":     result["model_a"],
            "Model_B":     result["model_b"],
            "chi2":        result["chi2_statistic"],
            "p_value":     result["p_value"],
            "Significant": result["significant"],
        })

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("p_value", ascending=True)
        .reset_index(drop=True)
    )

    print("\nMcNemar Test Summary")
    print(summary_df.to_string(index=False))

    return summary_df
