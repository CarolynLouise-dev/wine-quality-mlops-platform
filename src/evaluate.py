"""Classification metrics and visual evaluation artifacts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def _positive_scores(model: Any, features: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if callable(getattr(model, "predict_proba", None)):
        probabilities = np.asarray(model.predict_proba(features))
        return probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(features))
    return predictions


def _save_confusion_matrix(matrix: np.ndarray, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(title="Confusion Matrix", xlabel="Predicted label", ylabel="True label")
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    artifact_dir: str | Path | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate any sklearn-compatible binary classifier."""
    predictions = np.asarray(model.predict(X_test))
    scores = _positive_scores(model, X_test, predictions)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.5

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    artifacts: dict[str, Any] = {"confusion_matrix": matrix.tolist()}

    if artifact_dir is not None:
        image_path = Path(artifact_dir) / "confusion_matrix.png"
        _save_confusion_matrix(matrix, image_path)
        artifacts["confusion_matrix_path"] = str(image_path)

    LOGGER.info("Model evaluation: %s", metrics)
    return metrics, artifacts
