# vi-ocr-bench

Benchmark OCR chữ viết tay tiếng Việt.

Repo cung cấp:

- Bộ metrics dùng chung: CER, WER, diacritic-insensitive CER, tone error rate.
- Hàm benchmark đơn giản để chạy model trên cùng một dataset.
- Các script mẫu cho một số model OCR.

Dataset mặc định: `5CD-AI/Viet-Handwriting-OCR-v2`.

## Cài Đặt

Tạo môi trường Python và cài package:

```bash
cd ./vi-ocr-bench
uv venv
source .venv/bin/activate
uv pip install -e ".[paddleocr,hf,chandra]"
```

Dataset trên Hugging Face cần đăng nhập trước khi chạy:

```bash
huggingface-cli login
```

Có thể chỉ cài extra cần dùng:

```bash
uv pip install -e ".[chandra]"
uv pip install -e ".[paddleocr]"
uv pip install -e ".[hf]"
```

Với PaddleOCR chạy GPU, cần cài đúng bản PaddlePaddle theo CUDA của máy. Ví dụ CUDA 12.6:

```bash
uv pip install "paddlepaddle-gpu==3.3.1" \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match
```

## Chạy Model Có Sẵn

Chandra OCR 2:

```bash
python model_scripts/chandra_ocr_2.py --max-samples 5
```

PaddleOCR v5:

```bash
python model_scripts/paddleocr_v5.py --max-samples 5
```

PaddleOCR-VL:

```bash
python model_scripts/paddleocr_vl.py --max-samples 5
```

Bỏ `--max-samples` để chạy toàn bộ split test.

## Thêm Model Mới

Cách đơn giản nhất là copy template:

```bash
cp model_scripts/template.py model_scripts/my_model.py
```

Trong file mới, chỉ cần viết phần load model và hàm `predict`.

```python
from benchmark import run_benchmark

model = YourModel.load("cuda:0")


def predict(image_path: str) -> str:
    return model.ocr(image_path)


run_benchmark("my-model", predict)
```

Chạy benchmark:

```bash
python model_scripts/my_model.py
```

Quy ước duy nhất: `predict(image_path)` nhận đường dẫn ảnh và trả về text OCR dạng `str`.

## Tự Điều Khiển Vòng Lặp

Trong trường hợp model cần xử lý đặc biệt, có thể tự load dataset và tự gọi evaluate:

```python
from benchmark import load_dataset, save_and_evaluate

samples = load_dataset(split="test", max_samples=10)

predictions = []
for sample in samples:
    image = sample["image"]
    image.save("/tmp/image.png")
    predictions.append(your_model("/tmp/image.png"))

references = [sample["text"] for sample in samples]
save_and_evaluate("my-model", predictions, references)
```

## Evaluate File Predictions

Evaluate lại một file predictions có sẵn:

```bash
python evaluate.py output/my-model/predictions_test.json
```

Hoặc dùng module `vi_metrics`:

```bash
python -m vi_metrics output/my-model/predictions_test.json
```

## Output

Mỗi lần chạy tạo thư mục:

```text
output/<model-name>/
```

Trong đó có:

- `predictions_test.json`: kết quả OCR của từng sample.
- `metrics_test.json`: điểm đánh giá tổng hợp.

Format predictions:

```json
{
  "model": "my-model",
  "dataset": "5CD-AI/Viet-Handwriting-OCR-v2",
  "split": "test",
  "predictions": [
    {
      "index": 0,
      "reference": "ground truth text",
      "prediction": "model output"
    }
  ]
}
```

## Metrics

| Metric | Ý Nghĩa |
| --- | --- |
| CER | Character Error Rate trên text gốc |
| WER | Word Error Rate trên text gốc |
| Diacritic-insensitive CER | CER sau khi bỏ toàn bộ dấu tiếng Việt và đổi `đ` thành `d` |
| Tone error rate | Tỷ lệ sai dấu thanh trên các nguyên âm tiếng Việt |

## Cấu Trúc Project

```text
vi-ocr-bench/
├── benchmark.py          # Load dataset, chạy benchmark, lưu output, tính metrics
├── evaluate.py           # CLI evaluate file predictions có sẵn
├── model_scripts/        # Mỗi file là một script chạy benchmark cho một model
│   ├── template.py
│   ├── chandra_ocr_2.py
│   ├── paddleocr_v5.py
│   └── paddleocr_vl.py
└── vi_metrics/           # Thư viện metrics dùng độc lập
    ├── core.py           # Logic tính CER, WER, tone error rate
    ├── io.py             # Đọc predictions và ghi metrics report
    └── cli.py            # CLI cho vi_metrics
```

## Nguyên Tắc Thiết Kế

- Metrics nằm riêng trong `vi_metrics/` để dùng lại độc lập.
- Mỗi model là một script Python riêng trong `model_scripts/`.
- Không cần registry, class base, config phức tạp hoặc framework runner.
- Code chạy model có thể sửa trực tiếp theo đặc thù output của từng model.
