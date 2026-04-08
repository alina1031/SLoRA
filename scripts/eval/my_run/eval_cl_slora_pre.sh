#!/bin/bash
set -e

TASK_PATH=data
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs


# your config
domains=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")
ORDER=ORDER1

# slora_post
# CKPT=llama3-8b-lora-r64a128

# slora_pre
CKPT=llama3-8b-lora-r64a128-max

if [[ "$CKPT" =~ "llama2" ]]; then
    CONV_MODE=llama2
elif [[ "$CKPT" =~ "llama3" ]]; then
    CONV_MODE=llama3
elif [[ "$CKPT" =~ "qwen1.5" ]]; then
    CONV_MODE=qwen
else
    CONV_MODE=qwen
fi


CKPT=${ORDER}-${CKPT}
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
DENOISED_MODE=slora_pre
MAX_TEST_ORDER=8

export CUDA_VISIBLE_DEVICES=0,1,2,3

for domain in "${domains[@]}"; do
    TEMP_OUTPUT_DIR="$LOGS_BASE_PATH/$CKPT/$DENOISED_MODE/order${MAX_TEST_ORDER}"

    if [ -f "$TEMP_OUTPUT_DIR/$domain/eval.log" ]; then
        echo "Evaluation exists for ${domain}, skipping..."
        continue
    fi

    echo "====== Processing domain: ${domain} ======"

    if ! bash ./scripts/rebuttal/my_slurm/eval_cl.sh \
        "$TASK_PATH" "$MODEL_BASE" "$MODEL_PATH" "$CKPT" "$LOGS_BASE_PATH" \
        "$domain" "$CONV_MODE" "$DENOISED_MODE" "$MAX_TEST_ORDER" ; then
        echo "!!!! Eval failed for ${domain}, continuing to next."
        continue
    fi

    echo "====== Done: ${domain} ======"
done




# for idx in "${!domains[@]}"; do
#     domain="${domains[$idx]}"

#     # 第1个任务测1~8，第2个任务测2~8，第3个任务测3~8，第4个任务测4~8
#     start_order=$((idx + 1))

#     for ((test_order=start_order; test_order<=MAX_TEST_ORDER; test_order++)); do
#         TEMP_OUTPUT_DIR="$LOGS_BASE_PATH/$CKPT/$DROP_MODE/order${test_order}"

#         if [ -f "$TEMP_OUTPUT_DIR/$domain/eval.log" ]; then
#             echo "Evaluation exists for ${domain}, TEST_ORDER=${test_order}, skipping..."
#             continue
#         fi

#         echo "====== Processing domain: ${domain} (TEST_ORDER=${test_order}) ======"

#         if ! bash ./scripts/eval/my_slurm/eval_cl.sh \
#             "$TASK_PATH" "$MODEL_BASE" "$MODEL_PATH" "$CKPT" "$LOGS_BASE_PATH" \
#             "$domain" "$CONV_MODE" "$DENOISED_MODE" "$test_order"; then
#             echo "!!!! Eval failed for ${domain}, TEST_ORDER=${test_order}), continuing to next."
#             continue
#         fi

#         echo "====== Done: ${domain}, TEST_ORDER=${test_order} ======"
#     done
# done

echo "All domains completed."