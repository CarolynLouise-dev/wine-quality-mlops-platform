"""Feature preparation for the binary wine-quality classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

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
TARGET_COL = "quality"


def _persist_arrays(
    directory: Path,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }.items():
        np.save(directory / f"{name}.npy", values)
    joblib.dump(scaler, directory / "scaler.joblib")


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, dict[str, Any]]:
    """Create a reproducible stratified split and train-only standardization."""
    required = [*FEATURE_NAMES, TARGET_COL]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    features = df.loc[:, FEATURE_NAMES].astype(float).to_numpy()
    target = (pd.to_numeric(df[TARGET_COL]) >= 6).astype(int).to_numpy()
    if np.unique(target).size < 2:
        raise ValueError("Target must contain both binary classes after conversion")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    metadata: dict[str, Any] = {
        "train_samples": int(len(x_train_scaled)),
        "test_samples": int(len(x_test_scaled)),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES.copy(),
        "test_size": test_size,
        "random_state": random_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    if output_dir is not None:
        _persist_arrays(
            Path(output_dir),
            x_train_scaled,
            x_test_scaled,
            y_train,
            y_test,
            scaler,
        )

    LOGGER.info("Prepared %d training and %d test rows", len(y_train), len(y_test))
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, metadata


if __name__ == "__main__":
    from src.ingestion import ingest_data
    from src.validation import validate_dataset

    root = Path(__file__).resolve().parents[1]
    raw, _ = ingest_data(root / "data/raw/winequality-red.csv")
    clean, _ = validate_dataset(raw, max_bad_fraction=0.3)
    *_, details = preprocess_data(clean, output_dir=root / "data/processed")
    print(f"Preprocessing completed: {details['train_samples']} train rows")
