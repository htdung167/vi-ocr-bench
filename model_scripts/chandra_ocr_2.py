"""Benchmark Chandra OCR 2.

Cài đặt:
    uv pip install -e ".[chandra]"

Chạy:
    python model_scripts/chandra_ocr_2.py
    python model_scripts/chandra_ocr_2.py --max-samples 5
"""

from __future__ import annotations

import argparse

import torch
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from benchmark import run_benchmark

MODEL_ID = "datalab-to/chandra-ocr-2"

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device-map", default="auto")
args = parser.parse_args()

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=args.device_map,
)
model.eval()
model.processor = AutoProcessor.from_pretrained(MODEL_ID)
model.processor.tokenizer.padding_side = "left"


def predict(image_path: str) -> str:
    batch = [
        BatchInputItem(
            image=Image.open(image_path),
            prompt_type="ocr_layout",
        )
    ]
    result = generate_hf(batch, model)[0]
    return parse_markdown(result.raw).strip()


run_benchmark("chandra-ocr-2", predict, max_samples=args.max_samples)
