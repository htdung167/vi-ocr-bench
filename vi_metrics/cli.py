"""CLI for evaluating saved OCR predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vi_metrics.io import evaluate_predictions_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese OCR predictions.")
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for predictions_path in args.predictions:
        output_path = _default_output_path(predictions_path, args.output_dir)
        report = evaluate_predictions_file(predictions_path, output_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nSaved metrics to {output_path}")


def _default_output_path(predictions_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir / f"metrics_{predictions_path.stem}.json"
    return predictions_path.with_name(f"metrics_{predictions_path.stem}.json")


if __name__ == "__main__":
    main()
