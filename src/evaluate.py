"""Evaluation utilities for binary classification models.

This module provides functions to compute classification metrics and
save evaluation results.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (0 = negative, 1 = positive).
        y_pred: Predicted labels (0 = negative, 1 = positive).

    Returns:
        Dictionary with keys: "accuracy", "precision", "recall", "f1", "kappa".
        All values are floats rounded to 4 decimal places.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
    }
    return metrics


def evaluate_splits(
    model_name: str,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate model on train and test splits.

    Args:
        model_name: Name of the model for identification.
        y_train: Ground truth labels for training set.
        pred_train: Predicted labels for training set.
        y_test: Ground truth labels for test set.
        pred_test: Predicted labels for test set.

    Returns:
        DataFrame with 2 rows (train, test) and columns:
        model, split, accuracy, precision, recall, f1, kappa.
    """
    train_metrics = evaluate(y_train, pred_train)
    test_metrics = evaluate(y_test, pred_test)

    rows = [
        {"model": model_name, "split": "train", **train_metrics},
        {"model": model_name, "split": "test", **test_metrics},
    ]
    return pd.DataFrame(rows)


def save_metrics(df: pd.DataFrame, path: str) -> None:
    """Save metrics DataFrame to CSV.

    Creates the output directory if it does not exist.

    Args:
        df: DataFrame containing metrics.
        path: Output file path (e.g., "outputs/metrics/metrics.csv").
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Minimal test with perfect predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_perfect = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    metrics = evaluate(y_true, y_pred_perfect)

    assert metrics["accuracy"] == 1.0, f"Expected 1.0, got {metrics['accuracy']}"
    assert metrics["precision"] == 1.0, f"Expected 1.0, got {metrics['precision']}"
    assert metrics["recall"] == 1.0, f"Expected 1.0, got {metrics['recall']}"
    assert metrics["f1"] == 1.0, f"Expected 1.0, got {metrics['f1']}"
    assert metrics["kappa"] == 1.0, f"Expected 1.0, got {metrics['kappa']}"

    # Test with all zeros (edge case for zero_division=0)
    y_pred_all_zero = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    metrics_zero = evaluate(y_true, y_pred_all_zero)
    assert metrics_zero["precision"] == 0.0
    assert metrics_zero["recall"] == 0.0
    assert metrics_zero["f1"] == 0.0

    # Test evaluate_splits
    df = evaluate_splits(
        "test_model",
        y_train=y_true,
        pred_train=y_pred_perfect,
        y_test=y_true,
        pred_test=y_pred_perfect,
    )
    assert df.shape == (2, 7)
    assert list(df.columns) == [
        "model",
        "split",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
    ]
    assert df.iloc[0]["split"] == "train"
    assert df.iloc[1]["split"] == "test"
    assert all(df["accuracy"] == 1.0)

    # Test save_metrics
    save_metrics(df, "outputs/metrics/test_metrics.csv")
    saved = pd.read_csv("outputs/metrics/test_metrics.csv")
    assert saved.shape == (2, 7)

    print("All tests passed!")