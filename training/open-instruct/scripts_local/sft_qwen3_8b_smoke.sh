#!/bin/bash
set -e

cd /data/dhsun/zhrong/projects/shellagent/training/open-instruct

export TMPDIR="/data/dhsun/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="/data/dhsun/Models/Qwen3-8B-Base"
OUTPUT_DIR="/data/dhsun/zhrong/checkpoints/qwen3_8b_sft_smoke"
LOG_DIR="/data/dhsun/zhrong/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name qwen3_8b_sft_smoke \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$MODEL_PATH" \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --dataset_mixer_list /data/dhsun/Data/shellagent-sft 1.0 \
    --dataset_mixer_list_config_names skill_tax_20260505_2.2k_combined_balanced_thinking_all \
    --add_bos \
    --gradient_checkpointing \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --chat_template_name tulu \
    --seed 42
