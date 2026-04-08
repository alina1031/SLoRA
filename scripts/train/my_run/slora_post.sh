#!/bin/bash
set -e
TASK_PATH=data
MODEL_PATH=checkpoints
LOGS_BASE_PATH=logs

# Your config
MODEL_BASE=LLMs/Meta-Llama-3-8B-Instruct
MODEL=llama3
TRAINING_DATA=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")

CKPT=${MODEL}-8b-lora-r64a128


CKPT=${ORDER}-${CKPT}
LOG_DIR=${LOGS_BASE_PATH}/${CKPT}
OUTPUT_DIR=${MODEL_PATH}/${CKPT}

mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
TRAIN_EPOCHS=(5 3 7 5 3 5 5 7)


for idx in "${!TRAINING_DATA[@]}"; do
    ORDER=$((idx + 1))
    TASK_NAME="${TRAINING_DATA[$idx]}"
    TASK_FILE="${TASK_PATH}/${TASK_NAME}/train.json"
    TEMP_OUTPUT_DIR=${OUTPUT_DIR}/order${ORDER}
    TRAIN_EPOCH=${TRAIN_EPOCHS[$idx]}
    LOG_FILE="${LOG_DIR}/train_task${ORDER}.log"

    if [ -f "$TEMP_OUTPUT_DIR/adapter_model.safetensors" ]; then
        echo "Checkpoint exists for ${TASK_NAME} (order ${ORDER}), skipping..."
        continue
    fi

    echo "Starting training for ${TASK_NAME} (order ${ORDER})..."

    bash ./scripts/train/my_slurm/slora_post.sh $MODEL_BASE $MODEL "$TASK_NAME" "$TASK_FILE" $TEMP_OUTPUT_DIR $ORDER $TRAIN_EPOCH &> $LOG_FILE 

    if [ $? -ne 0 ]; then
        echo "ERROR: Training for ${TASK_NAME} failed (order ${ORDER}). Aborting sequence."
        exit 1
    fi
    echo "Completed training for ${TASK_NAME} (order ${ORDER}). Output in ${TEMP_OUTPUT_DIR}"

done

echo "All continuous learning tasks finished."