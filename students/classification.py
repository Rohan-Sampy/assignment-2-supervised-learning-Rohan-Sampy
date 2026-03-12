"""
Classification functions for logistic regression and k-nearest neighbors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def train_logistic_regression_grid(X_train, y_train, param_grid=None):
    """
    Train logistic regression models with grid search over hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

    model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=None,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train_knn_grid(X_train, y_train, param_grid=None):
    """
    Train k-NN models with grid search over hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=None,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def get_best_logistic_regression(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best logistic regression model with test-set evaluation.
    """
    grid_search = train_logistic_regression_grid(X_train, y_train, param_grid=param_grid)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_results_df': pd.DataFrame(grid_search.cv_results_),
        'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_auprc': float(average_precision_score(y_test, y_pred_proba)),
        'test_predictions': y_pred,
        'test_probabilities': y_pred_proba,
    }


def get_best_knn(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best k-NN model with test-set evaluation.
    """
    grid_search = train_knn_grid(X_train, y_train, param_grid=param_grid)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_k': int(grid_search.best_params_.get('n_neighbors')),
        'cv_results_df': pd.DataFrame(grid_search.cv_results_),
        'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_auprc': float(average_precision_score(y_test, y_pred_proba)),
        'test_predictions': y_pred,
        'test_probabilities': y_pred_proba,
    }


