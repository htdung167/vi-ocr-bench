"""Evaluate existing prediction files.

Usage:
    python evaluate.py output/my-model/predictions_test.json
    python evaluate.py file1.json file2.json file3.json
"""

from __future__ import annotations

import sys

from vi_metrics.io import evaluate_predictions_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <predictions.json> [more files...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        metrics_path = path.replace("predictions_", "metrics_")
        report = evaluate_predictions_file(path, metrics_path)
        print(f"\n{path}:")
        m = report["metrics"]
        print(f"  CER:  {m['cer']:.4f}")
        print(f"  WER:  {m['wer']:.4f}")
        print(f"  Diacritic-insensitive CER: {m['diacritic_insensitive_cer']:.4f}")
        print(f"  Tone error rate: {m['tone_error_rate']:.4f}")
        print(f"  Saved -> {metrics_path}")


if __name__ == "__main__":
    main()
