"""Train sklearn estimators and track their lifecycle with MLflow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.evaluate import evaluate_model

LOGGER = logging.getLogger(__name__)

EstimatorFactory = Callable[..., Any]
ESTIMATORS: dict[str, EstimatorFactory] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "svc": SVC,
}


def build_estimator(algo_name: str, params: dict[str, Any]) -> Any:
    """Build one of the supported classifiers from an explicit registry."""
    algorithm = algo_name.strip().lower()
    try:
        factory = ESTIMATORS[algorithm]
    except KeyError as exc:
        supported = ", ".join(sorted(ESTIMATORS))
        raise ValueError(f"Unsupported algorithm: {algo_name}. Choose from: {supported}") from exc

    options = dict(params)
    if algorithm in {"svm", "svc"}:
        options.setdefault("probability", True)
    return factory(**options)


def _configure_tracking(experiment_name: str, tracking_uri: str | None):
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def train_and_track_experiment(
    algo_name: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "wine_quality_experiment",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    register_model_name: str | None = None,
    artifact_dir: str | Path = "data/processed/artifacts",
) -> tuple[Any, dict[str, float], str]:
    """Fit a model, evaluate it and persist the complete MLflow run."""
    mlflow = _configure_tracking(experiment_name, tracking_uri)
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    model = build_estimator(algo_name, params)
    model.fit(X_train, y_train)

    title = run_name or f"{algo_name}_{params.get('random_state', 42)}"
    with mlflow.start_run(run_name=title) as run:
        mlflow.log_params(params)
        mlflow.set_tags({"algorithm": algo_name, "dataset": "winequality-red"})

        metrics, artifacts = evaluate_model(model, X_test, y_test, artifact_dir)
        mlflow.log_metrics(metrics)
        confusion_matrix_path = artifacts.get("confusion_matrix_path")
        if confusion_matrix_path:
            mlflow.log_artifact(confusion_matrix_path)

        sample = X_train[: min(5, len(X_train))]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
        )

        run_id = run.info.run_id
        if register_model_name:
            try:
                mlflow.register_model(f"runs:/{run_id}/model", register_model_name)
            except Exception as exc:  # registry is optional for local file tracking
                LOGGER.warning("Model registry unavailable: %s", exc)

    LOGGER.info("Completed %s (%s): %s", title, run_id, metrics)
    return model, metrics, run_id


def promote_best_model(
    model_name: str,
    target_stage: str = "Production",
    metric_name: str = "f1_score",
    experiment_name: str = "wine_quality_experiment",
    tracking_uri: str | None = None,
) -> str | None:
    """Promote the registered version attached to the best experiment run."""
    if tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
    else:
        import mlflow
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            LOGGER.warning("Experiment %s does not exist", experiment_name)
            return None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        best_run_id = runs[0].info.run_id
        version = next(
            (
                item.version
                for item in client.search_model_versions(f"name='{model_name}'")
                if item.run_id == best_run_id
            ),
            None,
        )
        if version is None:
            LOGGER.warning("Best run %s has no registered model version", best_run_id)
            return None

        try:
            client.set_registered_model_alias(model_name, target_stage.lower(), version)
        except Exception as exc:
            LOGGER.info("Registry alias is not supported by this backend: %s", exc)
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage,
                archive_existing_versions=True,
            )
        except Exception as exc:
            LOGGER.info("Legacy model stages are not supported by this backend: %s", exc)

        LOGGER.info("Promoted %s version %s to %s", model_name, version, target_stage)
        return str(version)
    except Exception as exc:
        LOGGER.warning("Could not promote a model: %s", exc)
        return None
