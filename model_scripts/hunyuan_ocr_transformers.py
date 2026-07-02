from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

from benchmark import run_benchmark

MODEL_ID = "models/tencent/HunyuanOCR"
PROMPT = "提取图中的文字。"

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device", default="cuda:0", help="VD: cuda:0, cuda:1")
args = parser.parse_args()

device = torch.device(args.device)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = HunYuanVLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=args.device,
)
model.eval()

generation_config = dict(
    max_new_tokens=2048,
    do_sample=False,
)


def clean_repeated_substrings(text: str) -> str:
    n = len(text)
    if n<1024:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text


def predict(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text_input, images=image, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)

    # Cắt bỏ phần input prompt, chỉ lấy phần generated
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return clean_repeated_substrings(output_text.strip())


run_benchmark("hunyuan-ocr-transformers", predict, max_samples=args.max_samples)
