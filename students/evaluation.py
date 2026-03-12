"""
Model evaluation functions: metrics and ROC/PR curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def calculate_r2_score(y_true, y_pred):
    """Calculate R² score for regression."""
    return float(r2_score(y_true, y_pred))


def calculate_classification_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_auroc_score(y_true, y_pred_proba):
    """Calculate Area Under the ROC Curve (AUROC)."""
    return float(roc_auc_score(y_true, y_pred_proba))


def calculate_auprc_score(y_true, y_pred_proba):
    """Calculate Area Under the Precision-Recall Curve (AUPRC)."""
    return float(average_precision_score(y_true, y_pred_proba))


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None, label=None):
    """
    Generate and plot ROC curve.
    """
    curve_label = label if label is not None else model_name
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, linewidth=2, label=f"{curve_label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if created_fig:
        fig.tight_layout()
    return fig


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None, label=None):
    """
    Generate and plot Precision-Recall curve.
    """
    curve_label = label if label is not None else model_name
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    baseline = float(np.mean(y_true))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(recall, precision, linewidth=2, label=f"{curve_label} (AUPRC={auprc:.3f})")
    ax.axhline(baseline, linestyle="--", color="gray", alpha=0.7, label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if created_fig:
        fig.tight_layout()
    return fig


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    generate_auroc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", ax=axes[0])
    generate_auroc_curve(y_true, y_pred_proba_knn, model_name="k-NN", ax=axes[0])
    axes[0].set_title("AUROC Comparison")

    generate_auprc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", ax=axes[1])
    generate_auprc_curve(y_true, y_pred_proba_knn, model_name="k-NN", ax=axes[1])
    axes[1].set_title("AUPRC Comparison")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
