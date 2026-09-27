"""Execute the reproducible ten-model MLflow experiment matrix."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "miniopassword")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

from src.ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.train import promote_best_model, train_and_track_experiment
from src.validation import validate_dataset

LOGGER = logging.getLogger("experiment-runner")


@dataclass(frozen=True)
class Experiment:
    name: str
    algorithm: str
    parameters: dict[str, Any]


EXPERIMENT_MATRIX = [
    Experiment("LogisticRegression_Baseline", "logistic_regression", {"C": 1.0, "max_iter": 500, "random_state": 42}),
    Experiment("LogisticRegression_L2_Strong", "logistic_regression", {"C": 0.1, "max_iter": 500, "random_state": 42}),
    Experiment("LogisticRegression_L2_Weak", "logistic_regression", {"C": 10.0, "max_iter": 500, "random_state": 42}),
    Experiment("RandomForest_Shallow_50Trees", "random_forest", {"n_estimators": 50, "max_depth": 5, "min_samples_split": 4, "random_state": 42}),
    Experiment("RandomForest_Medium_100Trees", "random_forest", {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": 42}),
    Experiment("RandomForest_Deep_200Trees", "random_forest", {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2, "random_state": 42}),
    Experiment("GradientBoosting_Slow_50Trees", "gradient_boosting", {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
    Experiment("GradientBoosting_Default_100Trees", "gradient_boosting", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42}),
    Experiment("GradientBoosting_Fast_150Trees", "gradient_boosting", {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 5, "random_state": 42}),
    Experiment("SVM_RBF_Kernel", "svm", {"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42}),
]


def resolve_tracking_uri(candidate_uri: str | None) -> str:
    """Use a reachable remote tracker, otherwise isolate runs in local storage."""
    candidate = candidate_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    if not candidate.startswith("http"):
        return candidate
    try:
        if requests.get(f"{candidate}/health", timeout=1.5).status_code == 200:
            return candidate
    except requests.RequestException:
        pass
    local_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
    LOGGER.info("MLflow at %s is unavailable; using %s", candidate, local_uri)
    return local_uri


def _prepare_dataset(data_path: str | Path | None):
    source = Path(data_path) if data_path else PROJECT_ROOT / "data/raw/winequality-red.csv"
    processed = PROJECT_ROOT / "data/processed"
    raw, _ = ingest_data(source, processed / "raw_snapshot.csv")
    clean, _ = validate_dataset(raw, quarantine_dir=processed / "validation")
    return preprocess_data(clean, output_dir=processed)[:4]


def run_experiment_matrix(
    data_path: str | Path | None = None,
    tracking_uri: str | None = None,
    register_best: bool = True,
) -> pd.DataFrame:
    """Train every planned configuration and return an F1-ranked table."""
    x_train, x_test, y_train, y_test = _prepare_dataset(data_path)
    uri = resolve_tracking_uri(tracking_uri)
    rows: list[dict[str, Any]] = []

    for position, experiment in enumerate(EXPERIMENT_MATRIX, start=1):
        LOGGER.info("[%d/%d] %s", position, len(EXPERIMENT_MATRIX), experiment.name)
        try:
            _, metrics, run_id = train_and_track_experiment(
                algo_name=experiment.algorithm,
                params=experiment.parameters,
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                run_name=experiment.name,
                tracking_uri=uri,
                register_model_name="wine_quality_model" if register_best else None,
                artifact_dir=PROJECT_ROOT / "data/processed/artifacts",
            )
            rows.append(
                {
                    "experiment_name": experiment.name,
                    "algorithm": experiment.algorithm,
                    "run_id": run_id[:8],
                    **metrics,
                }
            )
        except Exception:
            LOGGER.exception("Experiment failed: %s", experiment.name)

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = results.sort_values("f1_score", ascending=False).reset_index(drop=True)
    results.to_csv(PROJECT_ROOT / "data/processed/experiment_results.csv", index=False)
    print("\n=== EXPERIMENT RESULTS MATRIX ===")
    print(results.to_string(index=False))
    LOGGER.info("Best model: %s (F1 %.4f)", results.iloc[0]["experiment_name"], results.iloc[0]["f1_score"])

    if register_best:
        promote_best_model(
            model_name="wine_quality_model",
            target_stage="Production",
            metric_name="f1_score",
            tracking_uri=uri,
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment_matrix()
