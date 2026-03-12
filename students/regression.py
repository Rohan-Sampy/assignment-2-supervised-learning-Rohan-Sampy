"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):
    """
    Train ElasticNet models over a grid of hyperparameters.
    """
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    y_train = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)

    results = []
    for l1_ratio in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            results.append(
                {
                    "l1_ratio": float(l1_ratio),
                    "alpha": float(alpha),
                    "r2_score": float(r2_score(y_train, y_pred)),
                    "model": model,
                }
            )

    return pd.DataFrame(results)


def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    """
    pivot_df = results_df.pivot(index="alpha", columns="l1_ratio", values="r2_score")
    pivot_df = pivot_df.reindex(index=list(alphas), columns=list(l1_ratios))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "R² Score"}, ax=ax)
    ax.set_xlabel("L1 Ratio")
    ax.set_ylabel("Alpha")
    ax.set_title("ElasticNet R² Scores")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    y_train = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)
    y_test = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else np.asarray(y_test)

    results_df = train_elasticnet_grid(X_train, y_train, l1_ratios, alphas)

    best_model = None
    best_l1_ratio = None
    best_alpha = None
    best_train_r2 = -np.inf
    best_test_r2 = -np.inf

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            train_r2 = r2_score(y_train, model.predict(X_train))
            test_r2 = r2_score(y_test, model.predict(X_test))

            if test_r2 > best_test_r2:
                best_model = model
                best_l1_ratio = float(l1_ratio)
                best_alpha = float(alpha)
                best_train_r2 = float(train_r2)
                best_test_r2 = float(test_r2)

    return {
        "model": best_model,
        "best_l1_ratio": best_l1_ratio,
        "best_alpha": best_alpha,
        "train_r2": best_train_r2,
        "test_r2": best_test_r2,
        "results_df": results_df,
    }
