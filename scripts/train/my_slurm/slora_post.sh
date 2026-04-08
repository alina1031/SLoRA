#!/bin/bash
set -e

GPUS_PER_NODE=2
NNODES=1  
MASTER_PORT=$((RANDOM % 101 + 20000)) 
MASTER_ADDR="127.0.0.1" 
export LOGLEVEL=INFO
export CUDA_VISIBLE_DEVICES="4,5" 

MODEL_PATH="$1" 
MODEL="$2"
DATASET="$3"
DATA_PATH="$4"
OUTPUT_DIR="$5"
ORDER="$6"
TRAIN_EPOCH="$7"

echo "output dir is ${OUTPUT_DIR}, ${MODEL_PATH}, ${DATASET}, ${DATA_PATH}, ORDER=${ORDER}"

python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT  \
    --rdzv_backend c10d \
    --node_rank 0 \
    src/train/cl_train.py \
    --bf16 True \
    --use_peft True \
    --lora_r 64 --lora_alpha 128 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --model $MODEL \
    --dataset_name $DATASET \
    --train_data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $TRAIN_EPOCH \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --report_to none \
    --task_id $ORDER
