# PaddleOCR PP-OCRv5_server_rec -> DONE
# PaddleOCR-VL -> DONE
# Chandra-OCR-2 -> DONE
# VietOCR -> Anh Khang

# tencent/HunyuanOCR (1B)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download tencent/HunyuanOCR --local-dir ./models/tencent/HunyuanOCR --exclude "original" --token $HF_TOKEN

# rednote-hilab/dots.mocr (3B)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download rednote-hilab/dots.mocr --local-dir ./models/rednote-hilab/dots.mocr --exclude "original" --token $HF_TOKEN

# 5CD-AI/Vintern-1B-v3_5 (1B)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download 5CD-AI/Vintern-1B-v3_5 --local-dir ./models/5CD-AI/Vintern-1B-v3_5 --exclude "original" --token $HF_TOKEN

# deepseek-ai/DeepSeek-OCR-2 (3B)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download deepseek-ai/DeepSeek-OCR-2 --local-dir ./models/deepseek-ai/DeepSeek-OCR-2 --exclude "original" --token $HF_TOKEN

# nanonets/Nanonets-OCR2-3B (3B)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nanonets/Nanonets-OCR2-3B --local-dir ./models/nanonets/Nanonets-OCR2-3B --exclude "original" --token $HF_TOKEN