"""
Wine Quality MLOps Pipeline - Airflow orchestration DAG.

Pipeline Stages:
  [ingest] ➔ [validate] ➔ [preprocess] ➔ [train_and_evaluate] ➔ [register_and_promote] ➔ [sync_drift_baseline]

Features:
  - TaskFlow API (@dag, @task)
  - Data quality gate checks with automatic quarantine
  - Automated MLflow experiment tracking and model registration
  - Failure alerting and exponential backoff retry mechanism
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)

# Base Paths
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/mlops"))
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "winequality-red.csv"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def failure_alert_callback(context: dict):
    """Callback triggered whenever an Airflow task fails."""
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id
    execution_date = context.get("execution_date")
    log.error(
        "ALERT: Task %s in DAG %s failed on %s! Notifying MLOps on-call channel.",
        task_instance.task_id, dag_id, execution_date
    )


@dag(
    dag_id="wine_quality_mlops_pipeline",
    description="End-to-End Wine Quality Pipeline with Quality Gates, MLflow & Evidently",
    schedule="@daily",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops-team",
        "depends_on_past": False,
        "retries": 3,
        "retry_delay": timedelta(seconds=15),
        "retry_exponential_backoff": True,
        "on_failure_callback": failure_alert_callback,
    },
    tags=["mlops", "wine-quality", "production-pipeline"],
)
def wine_quality_mlops_pipeline():
    """Defines the sequential ML pipeline tasks and dependency flow."""

    @task
    def ingest_task(ds: str = None) -> dict:
        """Stage 1: Ingest raw data, compute SHA-256 integrity hash, and snapshot."""
        from src.ingestion import ingest_data

        if not RAW_DATA_PATH.exists():
            raise AirflowFailException(f"Raw source dataset missing: {RAW_DATA_PATH}")

        run_dir = STAGING_DIR / (ds or "default")
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = run_dir / "raw_snapshot.csv"

        _, meta = ingest_data(RAW_DATA_PATH, output_path=snapshot_file)
        log.info("Ingestion complete. Rows: %d, Hash: %s", meta["rows"], meta["sha256"][:12])
        return meta

    @task
    def validate_task(meta: dict, ds: str = None) -> dict:
        """Stage 2: Schema validation, domain bounds, and bad data quarantine."""
        import pandas as pd
        from src.validation import validate_dataset, DataValidationError

        snapshot_file = Path(meta["snapshot_path"])
        df = pd.read_csv(snapshot_file)

        quarantine_dir = STAGING_DIR / (ds or "default") / "validation"
        try:
            _, report = validate_dataset(df, max_bad_fraction=0.20, quarantine_dir=quarantine_dir)
        except DataValidationError as err:
            log.error("Quality gate violated: %s", err)
            raise AirflowFailException(f"Data validation failed: {err}") from err

        log.info("Validation passed. Clean rows: %d, Bad fraction: %.2f%%",
                 report["clean_rows"], report["bad_fraction"] * 100)
        report["clean_data_path"] = str(quarantine_dir / "clean_data.csv")
        return report

    @task
    def preprocess_task(val_report: dict, ds: str = None) -> dict:
        """Stage 3: Stratified split, feature scaling, and artifact saving."""
        import pandas as pd
        from src.preprocessing import preprocess_data

        clean_csv = Path(val_report["clean_data_path"])
        clean_df = pd.read_csv(clean_csv)

        out_dir = PROCESSED_DIR / (ds or "default")
        _, _, _, _, _, prep_meta = preprocess_data(
            clean_df, test_size=0.2, random_state=42, output_dir=out_dir
        )
        prep_meta["processed_dir"] = str(out_dir)
        log.info("Preprocessed %d train and %d test samples.",
                 prep_meta["train_samples"], prep_meta["test_samples"])
        return prep_meta

    @task
    def train_task(prep_meta: dict) -> dict:
        """Stage 4: Train candidate model, log parameters, metrics and signature to MLflow."""
        import numpy as np
        from src.train import train_and_track_experiment

        proc_dir = Path(prep_meta["processed_dir"])
        X_train = np.load(proc_dir / "X_train.npy")
        X_test = np.load(proc_dir / "X_test.npy")
        y_train = np.load(proc_dir / "y_train.npy")
        y_test = np.load(proc_dir / "y_test.npy")

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42
        }

        _, metrics, run_id = train_and_track_experiment(
            algo_name="random_forest",
            params=model_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            experiment_name="wine_quality_pipeline",
            run_name="Airflow_Automated_Run",
            tracking_uri=tracking_uri,
            register_model_name="wine_quality_model",
            artifact_dir=proc_dir / "artifacts"
        )

        return {"run_id": run_id, "metrics": metrics}

    @task
    def gate_and_promote_task(train_result: dict) -> str:
        """Stage 5: Quality evaluation gate. If F1-score >= 0.70, promote to Production."""
        from src.train import promote_best_model

        metrics = train_result["metrics"]
        f1 = metrics.get("f1_score", 0.0)
        min_threshold = 0.70

        if f1 < min_threshold:
            raise AirflowFailException(
                f"Model gate failed: F1-score {f1:.4f} is below minimum threshold {min_threshold}"
            )

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        version = promote_best_model(
            model_name="wine_quality_model",
            target_stage="Production",
            metric_name="f1_score",
            experiment_name="wine_quality_pipeline",
            tracking_uri=tracking_uri
        )
        log.info("Model passed quality gate! Promoted version: %s", version)
        return str(version or "unknown")

    @task
    def sync_drift_baseline_task(val_report: dict) -> bool:
        """Stage 6: Synchronize clean reference data with Evidently Drift Service."""
        import requests
        import pandas as pd

        evidently_url = os.getenv("EVIDENTLY_SERVICE_URL", "http://evidently:8001")
        clean_csv = Path(val_report["clean_data_path"])
        df = pd.read_csv(clean_csv)

        payload = {
            "data": df.to_dict(orient="records"),
            "feature_names": [c for c in df.columns if c != "quality"],
            "description": f"Baseline synchronized via Airflow at {datetime.utcnow().isoformat()}"
        }

        try:
            resp = requests.post(f"{evidently_url}/reference", json=payload, timeout=10)
            if resp.status_code == 200:
                log.info("Evidently reference baseline successfully updated.")
                return True
            log.warning("Evidently returned status: %d - %s", resp.status_code, resp.text)
            return False
        except Exception as exc:
            log.warning("Could not reach Evidently service: %s. Continuing pipeline.", exc)
            return False

    # Pipeline task dependencies
    ingest_meta = ingest_task()
    val_rep = validate_task(ingest_meta)
    prep_meta = preprocess_task(val_rep)
    train_res = train_task(prep_meta)
    promoted_ver = gate_and_promote_task(train_res)
    sync_drift = sync_drift_baseline_task(val_rep)

    promoted_ver >> sync_drift


pipeline = wine_quality_mlops_pipeline()
