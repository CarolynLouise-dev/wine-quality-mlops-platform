"""Dataset schema checks, domain rules, quarantine output and quality gates."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "fixed_acidity": (2.0, 20.0),
    "volatile_acidity": (0.05, 2.0),
    "citric_acid": (0.0, 1.5),
    "residual_sugar": (0.5, 30.0),
    "chlorides": (0.005, 1.0),
    "free_sulfur_dioxide": (1.0, 100.0),
    "total_sulfur_dioxide": (5.0, 350.0),
    "density": (0.97, 1.05),
    "ph": (2.5, 4.5),
    "sulphates": (0.2, 2.5),
    "alcohol": (7.0, 17.0),
    "quality": (3.0, 10.0),
}


class DataValidationError(ValueError):
    """Raised when a dataset cannot pass its configured quality gate."""


def _issue_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    issues = pd.DataFrame(index=frame.index)
    issues["has_null"] = frame.isna().any(axis=1)

    for column, (lower, upper) in FEATURE_BOUNDS.items():
        if column not in frame:
            issues[f"{column}_missing"] = True
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        issues[f"{column}_not_numeric"] = numeric.isna() & frame[column].notna()
        issues[f"{column}_out_of_bounds"] = numeric.notna() & ~numeric.between(lower, upper)

    issues["is_duplicate"] = frame.duplicated(keep="first")
    return issues


def _write_quarantine(
    directory: str | Path,
    clean: pd.DataFrame,
    rejected: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    clean.to_csv(target / "clean_data.csv", index=False)
    rejected.to_csv(target / "rejected_data.csv", index=False)
    (target / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def validate_dataset(
    df: pd.DataFrame,
    max_bad_fraction: float = 0.20,
    quarantine_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return accepted rows and a report, or fail when rejection is excessive."""
    if not 0.0 <= max_bad_fraction <= 1.0:
        raise ValueError("max_bad_fraction must be between 0 and 1")
    if df.empty:
        raise DataValidationError("Input dataset is empty.")

    issues = _issue_matrix(df)
    rejected_mask = issues.any(axis=1)
    rejected_count = int(rejected_mask.sum())
    rejected_fraction = rejected_count / len(df)
    clean = df.loc[~rejected_mask].copy()
    rejected = df.loc[rejected_mask].copy()
    issue_counts = issues.sum().astype(int)

    report: dict[str, Any] = {
        "total_rows": int(len(df)),
        "clean_rows": int(len(clean)),
        "rejected_rows": rejected_count,
        "bad_fraction": round(float(rejected_fraction), 4),
        "max_allowed_bad_fraction": max_bad_fraction,
        "issues_breakdown": {
            name: int(count) for name, count in issue_counts.items() if count
        },
        "status": "PASSED" if rejected_fraction <= max_bad_fraction else "FAILED",
    }

    if quarantine_dir is not None:
        _write_quarantine(quarantine_dir, clean, rejected, report)

    LOGGER.info(
        "Validation %s: %d accepted, %d rejected",
        report["status"],
        len(clean),
        rejected_count,
    )
    if rejected_fraction > max_bad_fraction:
        raise DataValidationError(
            "Dataset failed quality gate: "
            f"{rejected_fraction:.2%} rejected (threshold: {max_bad_fraction:.2%})"
        )
    return clean, report


if __name__ == "__main__":
    from src.ingestion import ingest_data

    root = Path(__file__).resolve().parents[1]
    raw, _ = ingest_data(root / "data/raw/winequality-red.csv")
    _, validation_report = validate_dataset(
        raw, quarantine_dir=root / "data/processed/validation"
    )
    print(json.dumps(validation_report, indent=2))
