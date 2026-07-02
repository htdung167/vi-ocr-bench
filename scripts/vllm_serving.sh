# PaddleOCR PP-OCRv5_server_rec -> DONE
# PaddleOCR-VL -> DONE
# Chandra-OCR-2 -> DONE
# VietOCR -> Anh Khang

# tencent/HunyuanOCR (1B)
CUDA_VISIBLE_DEVICES=0 vllm serve models/tencent/HunyuanOCR \
    --port 8089 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.8

# # rednote-hilab/dots.mocr (3B)
# CUDA_VISIBLE_DEVICES=0 vllm serve models/rednote-hilab/dots.mocr \
#                     --port 8089 \
#                     --gpu-memory-utilization 0.9 \
#                     --chat-template-content-format string \
#                     --trust-remote-code

# # 5CD-AI/Vintern-1B-v3_5 (1B)
# CUDA_VISIBLE_DEVICES=0 vllm serve models/5CD-AI/Vintern-1B-v3_5 \
#                     --port 8089 \
#                     --gpu-memory-utilization 0.9 \
#                     --trust-remote-code

# deepseek-ai/DeepSeek-OCR-2 (3B)
# CUDA_VISIBLE_DEVICES=0 vllm serve models/deepseek-ai/DeepSeek-OCR-2 \
#     --port 8089 \
#     --gpu-memory-utilization 0.9 \
#     --trust-remote-code \
#     --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
#     --no-enable-prefix-caching \
#     --mm-processor-cache-gb 0

# nanonets/Nanonets-OCR2-3B (3B)
# CUDA_VISIBLE_DEVICES=0 vllm serve models/nanonets/Nanonets-OCR2-3B \
#     --port 8089 \
#     --gpu-memory-utilization 0.92 \
#     --limit-mm-per-prompt '{"image": 1}' \
#     --mm-processor-kwargs '{"max_pixels": 1003520}' \
#     --max-model-len 8192 \
#     --max-num-seqs 32 \
#     --enable-prefix-caching \
#     --trust-remote-code