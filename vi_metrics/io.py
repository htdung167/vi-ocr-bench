"""Prediction-file loading and metric-report writing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vi_metrics.core import MetricScores, evaluate_all


def load_predictions(path: Path | str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Load prediction JSON and return metadata, references, predictions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, list):
        meta: dict[str, Any] = {}
        records = payload
    else:
        meta = {key: value for key, value in payload.items() if key != "predictions"}
        records = payload.get("predictions", [])

    references = [record["reference"] for record in records]
    predictions = [record["prediction"] for record in records]
    return meta, references, predictions


def build_report(
    scores: MetricScores,
    *,
    predictions_file: Path | str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metrics": {
            "cer": round(scores.cer, 6),
            "wer": round(scores.wer, 6),
            "diacritic_insensitive_cer": round(scores.diacritic_insensitive_cer, 6),
            "tone_error_rate": round(scores.tone_error_rate, 6),
        },
        "counts": {
            "num_samples": scores.num_samples,
            "num_ref_chars": scores.num_ref_chars,
            "num_ref_words": scores.num_ref_words,
            "num_tone_positions": scores.num_tone_positions,
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if predictions_file is not None:
        report["predictions_file"] = str(predictions_file)
    if meta:
        report.update(meta)
    return report


def save_report(report: dict[str, Any], path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def evaluate_predictions_file(
    predictions_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    source = Path(predictions_path)
    meta, references, predictions = load_predictions(source)
    scores = evaluate_all(references, predictions)
    report = build_report(scores, predictions_file=source, meta=meta)

    if output_path is not None:
        save_report(report, output_path)
    return report
