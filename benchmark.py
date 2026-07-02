"""vi-ocr-bench: Vietnamese OCR Benchmark

Cách dùng đơn giản nhất (sync):

    from benchmark import run_benchmark

    model = load_my_model("cuda:0")

    def predict(image_path):
        return model.ocr(image_path)

    run_benchmark("my-model", predict)

Async cho các model serving qua API:

    import asyncio
    from benchmark import run_benchmark_async

    async def predict_async(image_path):
        return await call_api(image_path)

    asyncio.run(run_benchmark_async("my-model", predict_async, max_concurrency=16))

Nếu cần kiểm soát nhiều hơn:

    from benchmark import load_dataset, save_and_evaluate

    samples = load_dataset(split="test")

    predictions = []
    for sample in samples:
        sample["image"].save("/tmp/img.png")
        predictions.append(my_model("/tmp/img.png"))

    references = [s["text"] for s in samples]
    save_and_evaluate("my-model", predictions, references)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Awaitable, Callable

from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from vi_metrics import evaluate_all

DATASET = "5CD-AI/Viet-Handwriting-OCR-v2"
OUTPUT_DIR = Path("output")


def load_dataset(
    split: str = "test",
    max_samples: int | None = None,
    dataset: str = DATASET,
) -> list:
    """Load benchmark dataset from HuggingFace.

    Returns list of samples, each with keys "image" (PIL Image) and "text" (str).
    """
    if max_samples is None:
        return list(hf_load_dataset(dataset, split=split))
    return list(hf_load_dataset(dataset, split=f"{split}[:{max_samples}]"))


def save_and_evaluate(
    model_name: str,
    predictions: list[str],
    references: list[str],
    split: str = "test",
    output_dir: Path | str = OUTPUT_DIR,
) -> dict:
    """Save predictions to JSON and print evaluation metrics. Returns metrics dict."""
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model_name,
        "dataset": DATASET,
        "split": split,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "predictions": [
            {"index": i, "reference": ref, "prediction": pred}
            for i, (ref, pred) in enumerate(zip(references, predictions))
        ],
    }
    predictions_path = out / f"predictions_{split}.json"
    predictions_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved {len(predictions)} predictions -> {predictions_path}")

    scores = evaluate_all(references, predictions)
    metrics = {
        "model": model_name,
        "metrics": {
            "cer": round(scores.cer, 6),
            "wer": round(scores.wer, 6),
            "diacritic_insensitive_cer": round(scores.diacritic_insensitive_cer, 6),
            "tone_error_rate": round(scores.tone_error_rate, 6),
        },
        "num_samples": scores.num_samples,
    }
    metrics_path = out / f"metrics_{split}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics -> {metrics_path}")
    print(json.dumps(metrics["metrics"], indent=2))
    return metrics


def run_benchmark(
    model_name: str,
    predict_fn: Callable[[str], str],
    *,
    split: str = "test",
    max_samples: int | None = None,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict:
    """Run full benchmark: load data, predict, evaluate.

    predict_fn: nhận image_path (str), trả về text (str).
    """
    samples = load_dataset(split=split, max_samples=max_samples)
    references = [s["text"] for s in samples]
    predictions = []

    with tempfile.TemporaryDirectory(prefix="vi_ocr_bench_") as tmp_dir:
        tmp = Path(tmp_dir)
        for i, sample in enumerate(tqdm(samples, desc=model_name, unit="img")):
            img_path = tmp / f"{i:06d}.png"
            sample["image"].save(img_path)
            predictions.append(predict_fn(str(img_path)))

    return save_and_evaluate(model_name, predictions, references, split, output_dir)


async def run_benchmark_async(
    model_name: str,
    predict_fn: Callable[[str], Awaitable[str]],
    *,
    split: str = "test",
    max_samples: int | None = None,
    max_concurrency: int = 16,
    output_dir: Path | str = OUTPUT_DIR,
) -> dict:
    """Run benchmark bất đồng bộ cho các model serving qua API.

    predict_fn: async function, nhận image_path (str), trả về text (str).
    max_concurrency: số request song song tối đa (default 16).
    """
    samples = load_dataset(split=split, max_samples=max_samples)
    references = [s["text"] for s in samples]
    n = len(samples)
    predictions: list[str | None] = [None] * n

    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = atqdm(total=n, desc=model_name, unit="img")

    async def _predict_one(idx: int, img_path: str) -> None:
        async with semaphore:
            predictions[idx] = await predict_fn(img_path)
            pbar.update(1)

    with tempfile.TemporaryDirectory(prefix="vi_ocr_bench_") as tmp_dir:
        tmp = Path(tmp_dir)
        # Lưu ảnh ra đĩa trước (sync, nhanh) rồi chạy predict song song
        img_paths: list[str] = []
        for i, sample in enumerate(samples):
            img_path = tmp / f"{i:06d}.png"
            sample["image"].save(img_path)
            img_paths.append(str(img_path))

        tasks = [_predict_one(i, p) for i, p in enumerate(img_paths)]
        await asyncio.gather(*tasks)

    pbar.close()
    return save_and_evaluate(model_name, predictions, references, split, output_dir)
