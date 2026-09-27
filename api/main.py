"""HTTP inference service for the wine-quality model."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger("wine-quality-api")

FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]
FEATURE_COUNT = len(FEATURE_NAMES)


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name: str = os.getenv("MODEL_NAME", "wine_quality_model")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")
    evidently_url: str = os.getenv("EVIDENTLY_SERVICE_URL", "http://evidently:8001")


SETTINGS = Settings()
MLFLOW_TRACKING_URI = SETTINGS.tracking_uri
MODEL_NAME = SETTINGS.model_name
MODEL_STAGE = SETTINGS.model_stage
EVIDENTLY_SERVICE_URL = SETTINGS.evidently_url

REQUEST_COUNT = Counter(
    "api_requests_total", "Total HTTP requests received", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total inference predictions completed",
    ["model_name", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model inference computation latency",
    ["model_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5),
)
PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of output prediction values (0=Normal, 1=Good)",
    ["model_name"],
    buckets=(0.0, 0.5, 1.0),
)
FEATURE_VALUE = Histogram(
    "model_feature_value",
    "Real-time distribution of input feature values",
    ["feature_name"],
    buckets=(-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
)
CURRENT_MODEL_VERSION = Gauge(
    "model_version_info", "Current active model version", ["model_name", "version"]
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds", "Time taken to load the model", ["model_name"]
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction exceptions",
    ["model_name", "error_type"],
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
                "feature_names": FEATURE_NAMES,
            }
        }
    )

    features: list[float] = Field(description="Eleven wine measurements")
    feature_names: list[str] | None = Field(default=None)


class BatchPredictionRequest(BaseModel):
    samples: list[list[float]] = Field(description="Wine feature vectors")


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float | None = None
    model_name: str
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


class HeuristicModel:
    """Deterministic readiness fallback used until MLflow has a promoted model."""

    def predict(self, values: np.ndarray) -> np.ndarray:
        return ((values[:, 10] >= 10.5) & (values[:, 1] <= 0.55)).astype(int)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        predictions = self.predict(values)
        good_probability = np.where(predictions == 1, 0.8, 0.2)
        return np.column_stack((1.0 - good_probability, good_probability))


class ModelManager:
    """Own the active model and isolate MLflow loading from request handlers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.model_name
        self.model_version = "none"
        self.model: Any | None = None
        self.load_time = 0.0
        self.load_model()

    def _tracking_server_ready(self) -> bool:
        if not self.settings.tracking_uri.startswith("http"):
            return True
        try:
            response = requests.get(f"{self.settings.tracking_uri}/health", timeout=1.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _load_registered_model(self) -> tuple[Any, str]:
        import mlflow
        import mlflow.pyfunc

        mlflow.set_tracking_uri(self.settings.tracking_uri)
        model_uri = f"models:/{self.model_name}/{self.settings.model_stage}"
        loaded = mlflow.pyfunc.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            self.model_name, stages=[self.settings.model_stage]
        )
        version = str(versions[0].version) if versions else "1"
        return loaded, version

    def load_model(self) -> bool:
        started = time.perf_counter()
        try:
            if not self._tracking_server_ready():
                raise ConnectionError("tracking server is unavailable")
            self.model, self.model_version = self._load_registered_model()
            LOGGER.info("Loaded %s version %s", self.model_name, self.model_version)
        except Exception as exc:
            LOGGER.info("Using local fallback model: %s", exc)
            self.model = HeuristicModel()
            self.model_version = "fallback-heuristic"

        self.load_time = time.perf_counter() - started
        CURRENT_MODEL_VERSION.labels(self.model_name, self.model_version).set(1)
        MODEL_LOAD_TIME.labels(self.model_name).set(self.load_time)
        return self.model is not None

    def predict(self, features: list[float]) -> tuple[int, float | None, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        values = np.asarray(features, dtype=float).reshape(1, -1)
        started = time.perf_counter()
        prediction = int(np.asarray(self.model.predict(values))[0])
        confidence: float | None = None
        predictor = getattr(self.model, "predict_proba", None)
        if callable(predictor):
            try:
                confidence = float(np.asarray(predictor(values))[0, 1])
            except (IndexError, TypeError, ValueError):
                confidence = None
        elapsed = time.perf_counter() - started

        PREDICTION_COUNT.labels(self.model_name, self.model_version).inc()
        PREDICTION_LATENCY.labels(self.model_name).observe(elapsed)
        PREDICTION_VALUE.labels(self.model_name).observe(prediction)
        return prediction, confidence, elapsed


app = FastAPI(
    title="Wine Quality Prediction API",
    description="Model serving with Prometheus metrics and a drift-monitoring hook.",
    version="3.0.0",
)
manager = ModelManager(SETTINGS)
APP_STARTED_AT = time.perf_counter()


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - started
        )
        raise
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(
        time.perf_counter() - started
    )
    return response


def _validate_features(features: list[float]) -> None:
    if len(features) != FEATURE_COUNT:
        PREDICTION_ERRORS.labels(manager.model_name, "invalid_feature_count").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FEATURE_COUNT} features, received {len(features)}",
        )
    if not np.isfinite(features).all():
        raise HTTPException(status_code=422, detail="Features must contain finite numbers")


def forward_sample_to_evidently(features: list[float], prediction: int) -> None:
    sample = dict(zip(FEATURE_NAMES, features))
    try:
        requests.post(
            f"{SETTINGS.evidently_url}/capture",
            json={
                "features": sample,
                "prediction": prediction,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_version": manager.model_version,
            },
            timeout=2,
        )
    except requests.RequestException as exc:
        LOGGER.debug("Drift service did not accept sample: %s", exc)


@app.get("/")
def index() -> dict[str, Any]:
    return {
        "service": "Wine Quality ML Inference Service",
        "status": "operational",
        "model": manager.model_name,
        "version": manager.model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics",
            "model_info": "/model/info",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    loaded = manager.model is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=manager.model_name,
        model_version=manager.model_version,
        uptime_seconds=round(time.perf_counter() - APP_STARTED_AT, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    _validate_features(payload.features)
    names = payload.feature_names or FEATURE_NAMES
    if len(names) != FEATURE_COUNT:
        raise HTTPException(status_code=422, detail="feature_names must contain exactly 11 names")
    for name, value in zip(names, payload.features):
        FEATURE_VALUE.labels(name).observe(value)

    try:
        prediction, confidence, elapsed = manager.predict(payload.features)
    except Exception as exc:
        PREDICTION_ERRORS.labels(manager.model_name, "inference_error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}") from exc

    background_tasks.add_task(forward_sample_to_evidently, payload.features, prediction)
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_name=manager.model_name,
        model_version=manager.model_version,
        latency_ms=round(elapsed * 1000, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
    results: list[dict[str, int | float | None]] = []
    for sample in payload.samples:
        _validate_features(sample)
        prediction, confidence, _ = manager.predict(sample)
        results.append({"prediction": prediction, "confidence": confidence})
    return {"batch_size": len(results), "results": results}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": manager.model_name,
        "model_version": manager.model_version,
        "features": FEATURE_NAMES,
        "load_time_seconds": manager.load_time,
        "stage": SETTINGS.model_stage,
        "tracking_uri": SETTINGS.tracking_uri,
    }
