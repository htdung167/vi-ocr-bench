from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer

from benchmark import run_benchmark

MODEL_ID = "models/deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\nFree OCR."

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device", default="cuda:0", help="VD: cuda:0, cuda:1")
parser.add_argument(
    "--attn-impl",
    default="eager",
    choices=["flash_attention_2", "eager"],
    help="Attention implementation (default: eager)",
)
args = parser.parse_args()

device = torch.device(args.device)

torch.cuda.set_device(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    _attn_implementation=args.attn_impl,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().to(device=device, dtype=torch.bfloat16)


def predict(image_path: str) -> str:
    result = model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=image_path,
        output_path=os.path.dirname(image_path),
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=False,
        eval_mode=True,
    )

    if not isinstance(result, str):
        raise RuntimeError(f"DeepSeek-OCR-2 returned {type(result).__name__}, expected str")
    return result.strip()


run_benchmark("deepseek-ocr-2-transformers", predict, max_samples=args.max_samples)
