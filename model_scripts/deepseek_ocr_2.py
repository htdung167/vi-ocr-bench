from __future__ import annotations

import argparse
import asyncio
import base64

from openai import AsyncOpenAI

from benchmark import run_benchmark_async

MODEL_ID = "models/deepseek-ai/DeepSeek-OCR-2"
PROMPT = "Free OCR."

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--max-concurrency", type=int, default=16)
parser.add_argument("--api-base", default="http://localhost:8089/v1")
args = parser.parse_args()

client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_base, timeout=3600)


async def predict(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    )
    return response.choices[0].message.content.strip()


asyncio.run(
    run_benchmark_async(
        "deepseek-ocr-2",
        predict,
        max_samples=args.max_samples,
        max_concurrency=args.max_concurrency,
    )
)
