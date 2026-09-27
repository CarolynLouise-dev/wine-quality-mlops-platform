"""Reusable HTTP traffic scenarios for the observability stack."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from simulations.data_generator import WineDataGenerator

LOGGER = logging.getLogger("traffic-scenarios")


class BaseScenario(ABC):
    name = "Base"

    def __init__(self, api_url: str = "http://localhost:8000/predict"):
        self.api_url = api_url
        self.generator = WineDataGenerator()

    @abstractmethod
    def make_sample(self) -> dict[str, float]:
        raise NotImplementedError

    def run(self, count: int = 50, delay_sec: float = 0.02) -> dict[str, Any]:
        LOGGER.info("Running %s with %d requests", self.name, count)
        latencies: list[float] = []
        successes = 0
        with requests.Session() as session:
            for _ in range(count):
                sample = self.make_sample()
                started = time.perf_counter()
                try:
                    response = session.post(
                        self.api_url,
                        json={"features": list(sample.values()), "feature_names": list(sample)},
                        timeout=2,
                    )
                    latencies.append(time.perf_counter() - started)
                    successes += int(response.status_code == 200)
                except requests.RequestException:
                    LOGGER.debug("Simulation request failed", exc_info=True)
                if delay_sec:
                    time.sleep(delay_sec)
        average = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "scenario": self.name,
            "total_sent": count,
            "successes": successes,
            "avg_latency_ms": round(average * 1000, 2),
        }


class NormalTrafficScenario(BaseScenario):
    name = "NormalTraffic"

    def make_sample(self) -> dict[str, float]:
        return self.generator.generate_normal_sample()


class SevereDriftScenario(BaseScenario):
    name = "SevereDrift"

    def make_sample(self) -> dict[str, float]:
        return self.generator.generate_drifted_sample(
            drift_factor=1.75,
            target_features=[
                "alcohol",
                "volatile_acidity",
                "sulphates",
                "chlorides",
                "total_sulfur_dioxide",
            ],
        )
