"""Synthetic wine samples for normal-traffic and drift scenarios."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")


class WineDataGenerator:
    def __init__(self, config_path: str | Path = CONFIG_PATH, seed: int | None = None):
        with Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.features_spec: dict[str, dict[str, float]] = config["features"]
        self.feature_names = list(self.features_spec)
        self.random = np.random.default_rng(seed)

    def _draw(self, feature: str, mean_factor: float = 1.0, std_factor: float = 1.0) -> float:
        specification = self.features_spec[feature]
        value = self.random.normal(
            specification["mean"] * mean_factor,
            specification["std"] * std_factor,
        )
        upper = specification["max"] * (1.3 if mean_factor != 1.0 else 1.0)
        return round(float(np.clip(value, specification["min"], upper)), 4)

    def generate_normal_sample(self) -> dict[str, float]:
        return {feature: self._draw(feature) for feature in self.features_spec}

    def generate_drifted_sample(
        self,
        drift_factor: float = 1.6,
        target_features: list[str] | None = None,
        noise_level: float = 0.2,
    ) -> dict[str, float]:
        sample = self.generate_normal_sample()
        targets = target_features or ["alcohol", "volatile_acidity", "sulphates", "chlorides"]
        for feature in targets:
            if feature in self.features_spec:
                sample[feature] = self._draw(feature, drift_factor, 1.0 + noise_level)
        return sample

    def generate_batch(
        self, count: int, drift: bool = False, drift_factor: float = 1.5
    ) -> list[dict[str, float]]:
        generator = (
            lambda: self.generate_drifted_sample(drift_factor=drift_factor)
            if drift
            else self.generate_normal_sample()
        )
        return [generator() for _ in range(count)]
