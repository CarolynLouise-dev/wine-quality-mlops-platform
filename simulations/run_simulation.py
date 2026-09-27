"""Run a stable-traffic phase followed by an intentional drift phase."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulations.scenarios import NormalTrafficScenario, SevereDriftScenario

LOGGER = logging.getLogger("simulation-runner")
API_URL = os.getenv("API_URL", "http://localhost:8000")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")


def check_services() -> bool:
    try:
        endpoints = (f"{API_URL}/health", f"{EVIDENTLY_URL}/health")
        return all(requests.get(url, timeout=2).status_code == 200 for url in endpoints)
    except requests.RequestException:
        return False


def analyze(window_size: int) -> dict[str, Any]:
    response = requests.post(
        f"{EVIDENTLY_URL}/analyze", json={"window_size": window_size}, timeout=15
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    if not check_services():
        LOGGER.error("API or drift service is unavailable; start docker compose first")
        return

    phases = [
        (NormalTrafficScenario(f"{API_URL}/predict"), 50),
        (SevereDriftScenario(f"{API_URL}/predict"), 60),
    ]
    for scenario, count in phases:
        traffic_result = scenario.run(count=count, delay_sec=0.01)
        LOGGER.info("Traffic result: %s", traffic_result)
        try:
            drift_result = analyze(count)
            LOGGER.info(
                "Drift=%s share=%s report=%s",
                drift_result.get("dataset_drift_detected"),
                drift_result.get("drift_share"),
                drift_result.get("html_report"),
            )
        except requests.RequestException as exc:
            LOGGER.warning("Analysis failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
