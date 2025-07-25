#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# NPROC_PER_NODE=1
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
NNODES=${WORLD_SIZE:-1}
# NNODES = 1
# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=pacs_material%100
# Output configuration
run_name="qwen2vl-3B-PACS-material-finetune"
output_dir=./output_m/

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --data_packing False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --fp16 \
    --output_dir ${output_dir} \
    --num_train_epochs 30 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --video_max_frames 20 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard" \
    
    

# Launch training
torchrun --nproc-per-node ${NPROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${entry_file} ${args}