"""Streaming data-drift service with Prometheus metrics and archived reports."""
from __future__ import annotations

import html
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from scipy.stats import ks_2samp

LOGGER = logging.getLogger("drift-monitor")
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", BASE_DIR / "reports"))
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", BASE_DIR / "reference"))
DRIFT_THRESHOLD = float(os.getenv("EVIDENTLY_DRIFT_THRESHOLD", "0.1"))
MIN_SAMPLES_FOR_ANALYSIS = int(os.getenv("EVIDENTLY_MIN_SAMPLES", "30"))
MAX_PRODUCTION_SAMPLES = 10_000

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected", "Overall data drift status (1=drift, 0=stable)"
)
DRIFT_SCORE = Gauge("evidently_drift_score", "Share of features with drift")
FEATURE_DRIFT = Gauge(
    "evidently_feature_drift", "Per-feature drift status", ["feature_name"]
)
DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_drifted_features_count", "Number of drifted features"
)
ANALYSIS_COUNT = Counter("evidently_analysis_total", "Drift analyses completed")
ANALYSIS_DURATION = Histogram(
    "evidently_analysis_duration_seconds",
    "Drift analysis duration",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)
MISSING_VALUES = Gauge(
    "evidently_missing_values_ratio", "Current missing-value ratio", ["feature_name"]
)


class SingleCapture(BaseModel):
    features: dict[str, float]
    prediction: int | None = None
    timestamp: str | None = None
    model_version: str | None = None


class BatchCapture(BaseModel):
    data: list[dict[str, Any]]


class ReferenceUpload(BaseModel):
    data: list[dict[str, Any]]
    feature_names: list[str] | None = None
    description: str | None = None


class AnalysisRequest(BaseModel):
    window_size: int = Field(default=100, gt=0)
    threshold: float = Field(default=DRIFT_THRESHOLD, ge=0.0, le=1.0)


class DataStore:
    """Thread-safe in-memory production window with a durable reference baseline."""

    def __init__(self) -> None:
        self.reference_data: pd.DataFrame | None = None
        self.production_samples: deque[dict[str, Any]] = deque(maxlen=MAX_PRODUCTION_SAMPLES)
        self.last_analysis_time: str | None = None
        self._lock = RLock()
        self._load_reference()

    def _load_reference(self) -> None:
        source = REFERENCE_DIR / "reference_data.csv"
        if source.is_file():
            try:
                self.reference_data = pd.read_csv(source)
            except (OSError, pd.errors.ParserError) as exc:
                LOGGER.error("Could not load reference dataset: %s", exc)

    def set_reference(self, frame: pd.DataFrame, description: str = "") -> None:
        with self._lock:
            self.reference_data = frame.copy()
            frame.to_csv(REFERENCE_DIR / "reference_data.csv", index=False)
            metadata = {
                "description": description,
                "rows": len(frame),
                "features": frame.columns.tolist(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (REFERENCE_DIR / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    def add_production(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.production_samples.append(dict(row))

    def get_production_df(self, window_size: int | None = None) -> pd.DataFrame:
        with self._lock:
            rows = list(self.production_samples)
        if window_size:
            rows = rows[-window_size:]
        return pd.DataFrame(rows)


store = DataStore()
app = FastAPI(
    title="Evidently AI Drift Monitoring Service",
    description="Statistical drift analysis, metrics and archived HTML reports.",
    version="3.0.0",
)


def _comparable_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    ignored = {"quality", "prediction", "captured_at", "model_version"}
    return [
        column
        for column in reference.columns
        if column in current.columns
        and column not in ignored
        and pd.api.types.is_numeric_dtype(reference[column])
    ]


def _column_drift(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        return {"drift_detected": False, "p_value": 1.0, "statistic": 0.0}
    result = ks_2samp(reference_values, current_values)
    return {
        "drift_detected": bool(result.pvalue < 0.05),
        "p_value": round(float(result.pvalue), 6),
        "statistic": round(float(result.statistic), 6),
    }


def _render_report(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{'DRIFT' if details['drift_detected'] else 'stable'}</td>"
        f"<td>{details['statistic']:.4f}</td>"
        f"<td>{details['p_value']:.6f}</td>"
        "</tr>"
        for name, details in summary["column_details"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Wine Data Drift Report</title>
<style>body{{font-family:system-ui;margin:40px;color:#18212f}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:10px;border:1px solid #d8dee9;text-align:left}}th{{background:#eef3f8}}
.status{{padding:16px;border-radius:8px;background:#f4f7fa;margin:20px 0}}</style></head>
<body><h1>Wine Data Drift Report</h1><div class="status">
<strong>Dataset drift:</strong> {summary['dataset_drift_detected']} &nbsp;|&nbsp;
<strong>Drift share:</strong> {summary['drift_share']:.2%} &nbsp;|&nbsp;
<strong>Window:</strong> {summary['window_size']} samples</div>
<table><thead><tr><th>Feature</th><th>Status</th><th>KS statistic</th><th>p-value</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""


def run_drift_calculation(window_size: int, threshold: float) -> dict[str, Any]:
    """Compare the latest production window with the stored reference distribution."""
    started = time.perf_counter()
    if store.reference_data is None:
        raise HTTPException(status_code=400, detail="Reference baseline dataset has not been initialized.")

    current = store.get_production_df(window_size)
    if len(current) < MIN_SAMPLES_FOR_ANALYSIS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient production samples ({len(current)}). "
                f"Minimum required is {MIN_SAMPLES_FOR_ANALYSIS}"
            ),
        )
    columns = _comparable_columns(store.reference_data, current)
    if not columns:
        raise HTTPException(status_code=400, detail="No matching numeric reference features were captured.")

    details = {
        column: _column_drift(store.reference_data[column], current[column])
        for column in columns
    }
    drifted = [column for column, result in details.items() if result["drift_detected"]]
    drift_share = len(drifted) / len(columns)
    dataset_drift = drift_share >= threshold if threshold > 0 else bool(drifted)

    for column, result in details.items():
        FEATURE_DRIFT.labels(column).set(int(result["drift_detected"]))
        MISSING_VALUES.labels(column).set(float(current[column].isna().mean()))
    DRIFT_DETECTED.set(int(dataset_drift))
    DRIFT_SCORE.set(drift_share)
    DRIFTED_FEATURES_COUNT.set(len(drifted))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(drift_share, 6),
        "number_of_drifted_columns": len(drifted),
        "drift_by_columns": {
            column: int(result["drift_detected"]) for column, result in details.items()
        },
        "column_details": details,
        "window_size": len(current),
    }
    html_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    json_path = REPORTS_DIR / f"drift_report_{timestamp}.json"
    html_path.write_text(_render_report(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    ANALYSIS_COUNT.inc()
    ANALYSIS_DURATION.observe(elapsed)
    store.last_analysis_time = timestamp
    return {
        key: value
        for key, value in {
            **summary,
            "html_report": html_path.name,
            "duration_seconds": round(elapsed, 3),
        }.items()
        if key not in {"column_details", "window_size"}
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "reference_loaded": store.reference_data is not None,
        "reference_samples": len(store.reference_data) if store.reference_data is not None else 0,
        "production_samples": len(store.production_samples),
        "last_analysis": store.last_analysis_time,
        "reports_count": len(list(REPORTS_DIR.glob("*.html"))),
    }


@app.post("/reference")
def set_reference(payload: ReferenceUpload) -> dict[str, Any]:
    if not payload.data:
        raise HTTPException(status_code=400, detail="Data list is empty.")
    frame = pd.DataFrame(payload.data)
    store.set_reference(frame, payload.description or "User uploaded reference")
    return {
        "message": "Reference baseline updated successfully.",
        "samples": len(frame),
        "features": frame.columns.tolist(),
    }


@app.get("/reference")
def get_reference() -> dict[str, Any]:
    if store.reference_data is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "samples": len(store.reference_data),
        "columns": store.reference_data.columns.tolist(),
    }


@app.post("/capture")
def capture(payload: SingleCapture) -> dict[str, Any]:
    row: dict[str, Any] = dict(payload.features)
    if payload.prediction is not None:
        row["prediction"] = payload.prediction
    row["captured_at"] = payload.timestamp or datetime.now(timezone.utc).isoformat()
    if payload.model_version:
        row["model_version"] = payload.model_version
    store.add_production(row)
    return {"status": "captured", "total_samples": len(store.production_samples)}


@app.post("/capture/batch")
def capture_batch(payload: BatchCapture) -> dict[str, Any]:
    for row in payload.data:
        store.add_production(row)
    return {"status": "batch_captured", "total_samples": len(store.production_samples)}


@app.post("/analyze")
def trigger_analysis(request: AnalysisRequest) -> dict[str, Any]:
    return run_drift_calculation(request.window_size, request.threshold)


@app.get("/reports")
def list_reports() -> dict[str, list[str]]:
    return {"reports": sorted((path.name for path in REPORTS_DIR.glob("*.html")), reverse=True)}


@app.get("/reports/{report_name}")
def get_report(report_name: str) -> HTMLResponse:
    if Path(report_name).name != report_name or not report_name.endswith(".html"):
        raise HTTPException(status_code=404, detail="Report not found")
    target = REPORTS_DIR / report_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return HTMLResponse(target.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
