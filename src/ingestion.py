"""Load a raw wine dataset and create a traceable, normalized snapshot."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
READ_CHUNK_SIZE = 1024 * 1024

EXPECTED_FEATURES = [
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
    "quality",
]


def compute_file_hash(file_path: str | Path) -> str:
    """Return a SHA-256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as source:
        for chunk in iter(lambda: source.read(READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_column(column: object) -> str:
    return str(column).strip().strip('"').lower().replace(" ", "_")


def _read_csv(source: Path) -> pd.DataFrame:
    """Read either the UCI semicolon format or an ordinary CSV file."""
    try:
        frame = pd.read_csv(source, sep=";")
        return frame if frame.shape[1] > 1 else pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc


def ingest_data(
    raw_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, normalize and optionally persist a raw-data snapshot."""
    source = Path(raw_path)
    if not source.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty: {source}")

    source_hash = compute_file_hash(source)
    frame = _read_csv(source)
    if frame.empty:
        raise ValueError(f"Raw dataset contains no records: {source}")

    frame.columns = [_normalize_column(column) for column in frame.columns]
    metadata: dict[str, Any] = {
        "source_path": str(source),
        "sha256": source_hash,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
    }

    if output_path is not None:
        snapshot = Path(output_path)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(snapshot, index=False)
        metadata["snapshot_path"] = str(snapshot)

    LOGGER.info("Ingested %d rows from %s (%s)", len(frame), source, source_hash[:12])
    return frame, metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    _, ingestion_metadata = ingest_data(
        project_root / "data/raw/winequality-red.csv",
        project_root / "data/processed/raw_snapshot.csv",
    )
    print(f"Successfully ingested {ingestion_metadata['rows']} rows.")
