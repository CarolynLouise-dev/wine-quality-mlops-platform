"""Capture reproducible test and live-endpoint output as terminal-style PNG evidence."""
from __future__ import annotations

import json
import os
import subprocess
import textwrap
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidences"
FONT_PATH = Path("/System/Library/Fonts/Menlo.ttc")


def request_json(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def request_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def terminal_image(title: str, lines: list[str], destination: Path) -> None:
    width = 1500
    margin = 42
    header_height = 72
    font = ImageFont.truetype(str(FONT_PATH), 23) if FONT_PATH.exists() else ImageFont.load_default()
    wrapped = [piece for line in lines for piece in (textwrap.wrap(line, width=104) or [""])]
    line_height = 34
    height = header_height + margin + line_height * len(wrapped) + margin
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, header_height), fill="#1f2937")
    for index, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        x = 26 + index * 34
        draw.ellipse((x, 24, x + 18, 42), fill=color)
    draw.text((width // 2, 35), title, fill="#e5e7eb", font=font, anchor="mm")
    y = header_height + margin
    for line in wrapped:
        color = "#86efac" if "PASS" in line or "200" in line else "#e5e7eb"
        draw.text((margin, y), line, fill=color, font=font)
        y += line_height
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def capture_tests() -> None:
    environment = {**os.environ, "MPLCONFIGDIR": "/tmp/mlops-matplotlib"}
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    visible_lines = [
        line
        for line in result.stdout.strip().splitlines()
        if not line.startswith(("rootdir:", "cachedir:"))
    ]
    output = ["$ .venv/bin/python -m pytest tests -q", *visible_lines]
    if result.stderr.strip():
        output.extend(result.stderr.strip().splitlines())
    output.append(f"exit_code={result.returncode}")
    terminal_image("MLOps test suite", output, EVIDENCE_DIR / "11_refactored_tests_passed.png")
    if result.returncode:
        raise SystemExit(result.returncode)


def capture_runtime() -> None:
    sample = {
        "features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    }
    api_health = request_json("http://127.0.0.1:8000/health")
    prediction = request_json("http://127.0.0.1:8000/predict", sample)
    drift_health = request_json("http://127.0.0.1:8001/health")
    metrics = [
        line
        for line in request_text("http://127.0.0.1:8000/metrics").splitlines()
        if line.startswith(("api_requests_total", "model_predictions_total"))
    ]
    captured_at = datetime.now(timezone.utc).isoformat()
    output = [
        f"captured_at={captured_at}",
        "$ GET http://127.0.0.1:8000/health",
        f"HTTP 200 {json.dumps(api_health)}",
        "$ POST http://127.0.0.1:8000/predict",
        f"HTTP 200 {json.dumps(prediction)}",
        "$ GET http://127.0.0.1:8001/health",
        f"HTTP 200 {json.dumps(drift_health)}",
        "$ GET http://127.0.0.1:8000/metrics",
        *metrics[:8],
    ]
    terminal_image("Live API verification", output, EVIDENCE_DIR / "12_refactored_runtime_proof.png")
    (EVIDENCE_DIR / "runtime_verification.json").write_text(
        json.dumps(
            {
                "captured_at": captured_at,
                "api_health": api_health,
                "prediction": prediction,
                "drift_health": drift_health,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    capture_tests()
    capture_runtime()
    print("Saved test and runtime evidence under evidences/")
