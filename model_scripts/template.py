"""Template - Copy file này và điền code model của bạn vào.

Chạy:
    python model_scripts/template.py
    python model_scripts/template.py --max-samples 5
"""

import argparse

from benchmark import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
args = parser.parse_args()

# === SETUP MODEL CỦA BẠN Ở ĐÂY ===
# model = YourModel.load("your-model-name", device="cuda:0")


def predict(image_path: str) -> str:
    """Nhận đường dẫn ảnh, trả về text OCR."""
    # === CODE PREDICT CỦA BẠN Ở ĐÂY ===
    # return model.predict(image_path)
    raise NotImplementedError("Điền code predict của bạn vào đây")


run_benchmark("your-model-name", predict, max_samples=args.max_samples)
