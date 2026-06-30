"""Benchmark PaddleOCR v5 (PP-OCR pipeline).

Cài đặt:
    uv pip install -e ".[paddleocr]"

Chạy:
    python model_scripts/paddleocr_v5.py
    python model_scripts/paddleocr_v5.py --max-samples 5   # test nhanh
"""

import argparse

from paddleocr import PaddleOCR

from benchmark import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device", default="gpu:0")
args = parser.parse_args()

ocr = PaddleOCR(
    lang="vi",
    ocr_version="PP-OCRv5",
    device=args.device,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def predict(image_path: str) -> str:
    results = ocr.predict(image_path)
    if not results:
        return ""
    data = results[0].json.get("res", results[0].json)
    texts = data.get("rec_texts", [])
    boxes = data.get("rec_boxes")
    if boxes and len(boxes) == len(texts):
        pairs = sorted(zip(boxes, texts), key=lambda p: p[0][0])
        texts = [t for _, t in pairs]
    return " ".join(t for t in texts if t)


run_benchmark("paddleocr-v5", predict, max_samples=args.max_samples)
