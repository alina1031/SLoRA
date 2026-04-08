#!/bin/bash
set -euo pipefail

TASK_PATH="$1"
MODEL_BASE="$2"
MODEL_PATH="$3"
CKPT="$4"
LOGS_BASE_PATH="$5"
DATASET="$6"
CONV_MODE="$7"
DENOISED_MODE="$8"
TEST_ORDER="$9"

DATA_PATH="${TASK_PATH}/${DATASET}"
MODEL_PATH="${MODEL_PATH}/${CKPT}"

OUT_DIR="${LOGS_BASE_PATH}/${CKPT}/${DENOISED_MODE}/order${TEST_ORDER}/${DATASET}"
mkdir -p "${OUT_DIR}"
exec >> "${OUT_DIR}/run.log" 2>&1

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Task path=${TASK_PATH}"
echo "[INFO] Model base=${MODEL_BASE}"
echo "[INFO] Model path=${MODEL_PATH}"
echo "[INFO] Dataset=${DATASET}"
echo "[INFO] CONV_MODE=${CONV_MODE}, DENOISED_MODE=${DROP_MODE}, TEST_ORDER=${TEST_ORDER}"
echo "[INFO] Logs -> ${OUT_DIR}"

echo "[RUN] Inference ${DATASET}"
python -u -m src.eval.model_diverse_gen_batch \
    --model-path "${MODEL_PATH}" \
    --model-base "${MODEL_BASE}" \
    --question-file "${DATA_PATH}/test.json" \
    --answers-file "${OUT_DIR}/infer.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}" \
    --denoised_mode "${DENOISED_MODE}" \
    --test_order "${TEST_ORDER}" \
    --batch_size 1 \
    --resume \
    >> "${OUT_DIR}/infer.log" 2>&1

echo "[RUN] Evaluate ${DATASET}"
python -u -m src.eval.eval_trace \
    --input_file "${OUT_DIR}/infer.jsonl" \
    --output_file "${OUT_DIR}/wrong.jsonl" \
    >> "${OUT_DIR}/eval.log" 2>&1
    