from __future__ import annotations

import argparse

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

from benchmark import run_benchmark

MODEL_ID = "models/rednote-hilab/dots.mocr"
PROMPT = "Extract the text content from this image."

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device", default="cuda:0", help="VD: cuda:0, cuda:1")
args = parser.parse_args()

device = torch.device(args.device)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=args.device,
)
model.eval()

generation_config = dict(
    max_new_tokens=1024,
    do_sample=False,
)


def predict(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_config)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


run_benchmark("dots-mocr-transformers", predict, max_samples=args.max_samples)
