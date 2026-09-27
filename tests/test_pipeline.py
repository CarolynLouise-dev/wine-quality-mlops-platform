"""
Unit Tests for Data Ingestion, Validation, Preprocessing, and Training Modules.
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.ingestion import ingest_data, compute_file_hash
from src.validation import validate_dataset, DataValidationError
from src.preprocessing import preprocess_data
from src.evaluate import evaluate_model
from src.train import build_estimator, train_and_track_experiment


@pytest.fixture
def sample_wine_df():
    """Generates a valid mock wine dataframe with 11 features and quality."""
    np.random.seed(42)
    n = 100
    data = {
        "fixed_acidity": np.random.uniform(5.0, 14.0, n),
        "volatile_acidity": np.random.uniform(0.15, 1.2, n),
        "citric_acid": np.random.uniform(0.0, 0.8, n),
        "residual_sugar": np.random.uniform(1.0, 10.0, n),
        "chlorides": np.random.uniform(0.02, 0.2, n),
        "free_sulfur_dioxide": np.random.uniform(5.0, 50.0, n),
        "total_sulfur_dioxide": np.random.uniform(10.0, 150.0, n),
        "density": np.random.uniform(0.992, 1.001, n),
        "ph": np.random.uniform(3.0, 3.8, n),
        "sulphates": np.random.uniform(0.4, 1.2, n),
        "alcohol": np.random.uniform(9.0, 14.0, n),
        "quality": np.random.choice([4, 5, 6, 7], n)
    }
    return pd.DataFrame(data)


def test_file_hash(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("mlops testing string")
    h1 = compute_file_hash(p)
    h2 = compute_file_hash(p)
    assert len(h1) == 64
    assert h1 == h2


def test_ingestion(tmp_path, sample_wine_df):
    raw_file = tmp_path / "wine.csv"
    sample_wine_df.to_csv(raw_file, sep=";", index=False)

    out_file = tmp_path / "snapshot.csv"
    df, meta = ingest_data(raw_file, output_path=out_file)

    assert len(df) == 100
    assert meta["rows"] == 100
    assert out_file.exists()


def test_validation_passes(sample_wine_df, tmp_path):
    clean, report = validate_dataset(sample_wine_df, max_bad_fraction=0.1, quarantine_dir=tmp_path)
    assert report["status"] == "PASSED"
    assert len(clean) > 90
    assert (tmp_path / "validation_report.json").exists()


def test_validation_rejects_corrupted(sample_wine_df):
    # Corrupt multiple rows with negative values
    corrupted = sample_wine_df.copy()
    corrupted.loc[:50, "alcohol"] = -99.0

    with pytest.raises(DataValidationError):
        validate_dataset(corrupted, max_bad_fraction=0.1)


def test_preprocessing(sample_wine_df, tmp_path):
    X_tr, X_te, y_tr, y_te, scaler, meta = preprocess_data(
        sample_wine_df, test_size=0.2, random_state=42, output_dir=tmp_path
    )

    assert X_tr.shape[1] == 11
    assert X_te.shape[1] == 11
    assert len(y_tr) == 80
    assert len(y_te) == 20
    assert set(np.unique(y_tr)).issubset({0, 1})
    assert (tmp_path / "scaler.joblib").exists()


def test_build_estimator():
    rf = build_estimator("random_forest", {"n_estimators": 10, "random_state": 42})
    assert hasattr(rf, "fit")

    lr = build_estimator("logistic_regression", {"C": 0.5})
    assert hasattr(lr, "predict")

    with pytest.raises(ValueError):
        build_estimator("unknown_algo", {})


def test_evaluation():
    from sklearn.dummy import DummyClassifier
    X = np.random.randn(50, 11)
    y = np.random.choice([0, 1], 50)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X, y)

    metrics, artifacts = evaluate_model(dummy, X, y)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in artifacts
