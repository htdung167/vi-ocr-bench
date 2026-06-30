"""Benchmark PaddleOCR-VL 1.6.

Cài đặt:
    uv pip install -e ".[paddleocr]"

Chạy:
    python model_scripts/paddleocr_vl.py
    python model_scripts/paddleocr_vl.py --max-samples 5
"""

import argparse

from paddleocr import PaddleOCRVL

from benchmark import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--device", default="gpu:0")
parser.add_argument("--version", default="v1.6")
args = parser.parse_args()

pipeline = PaddleOCRVL(
    pipeline_version=args.version,
    device=args.device,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_layout_detection=False,
)


def predict(image_path: str) -> str:
    results = pipeline.predict(
        image_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
    )
    if not results:
        return ""
    data = results[0].json.get("res", results[0].json)
    blocks = data.get("parsing_res_list", [])
    blocks = sorted(blocks, key=lambda b: (b.get("block_order", 0), b.get("block_id", 0)))
    return " ".join(b["block_content"] for b in blocks if b.get("block_content"))


run_benchmark(f"paddleocr-vl-{args.version}", predict, max_samples=args.max_samples)
