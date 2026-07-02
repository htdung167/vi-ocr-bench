from __future__ import annotations

import argparse
import asyncio
import base64
import re

from openai import AsyncOpenAI

from benchmark import run_benchmark_async

MODEL_ID = "models/nanonets/Nanonets-OCR2-3B"
PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, add a small "
    "description of the image inside the <img></img> tag; otherwise, add the image caption "
    "inside <img></img>. Watermarks should be wrapped in brackets. "
    "Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
    "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--max-concurrency", type=int, default=16)
parser.add_argument("--api-base", default="http://localhost:8089/v1")
args = parser.parse_args()

client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_base, timeout=3600)


def markdown_to_plain_text(md: str) -> str:
    """Convert markdown output to plain text."""
    text = md

    text = re.sub(r"<img>(.*?)</img>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<watermark>(.*?)</watermark>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<page_number>(.*?)</page_number>", r"\1", text, flags=re.DOTALL)

    text = re.sub(r"```[a-zA-Z]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)

    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)

    text = re.sub(r"~~(.*?)~~", r"\1", text)

    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)

    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    text = re.sub(r"^[\s]*([-*_]){3,}\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


async def predict(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]

    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    return markdown_to_plain_text(raw)


asyncio.run(
    run_benchmark_async(
        "nanonets-ocr2-3b",
        predict,
        max_samples=args.max_samples,
        max_concurrency=args.max_concurrency,
    )
)
