from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

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
    default="flash_attention_2",
    choices=["flash_attention_2", "eager"],
    help="Attention implementation (default: flash_attention_2)",
)
args = parser.parse_args()

device = torch.device(args.device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    _attn_implementation=args.attn_impl,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().to(device).to(torch.bfloat16)

generation_config = dict(
    max_new_tokens=2048,
    do_sample=False,
)


def predict(image_path: str) -> str:
    result = model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=image_path,
        output_path="test",
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=False,
    )

    if isinstance(result, list):
        return "\n".join(str(r) for r in result).strip()
    return str(result).strip()


run_benchmark("deepseek-ocr-2-transformers", predict, max_samples=args.max_samples)
