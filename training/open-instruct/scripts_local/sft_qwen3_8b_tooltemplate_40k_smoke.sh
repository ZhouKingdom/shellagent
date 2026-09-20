#!/bin/bash
set -e

cd /data/dhsun/zhrong/projects/shellagent/training/open-instruct

export TMPDIR="/data/dhsun/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export TRITON_CACHE_DIR="/data/dhsun/zhrong/.triton"
export HF_HOME="/data/dhsun/zhrong/.cache/huggingface"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

MODEL_PATH="/data/dhsun/Models/Qwen3-8B"
TOKENIZER_PATH="/data/dhsun/Models/shellagent-sft-8b"
OUTPUT_DIR="/data/dhsun/zhrong/checkpoints/qwen3_8b_sft_tooltemplate_32k_smoke"
LOG_DIR="/data/dhsun/zhrong/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name qwen3_8b_sft_tooltemplate_32k_smoke \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --chat_template_name "$TOKENIZER_PATH" \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --max_train_steps 5 \
    --dataset_mixer_list /data/dhsun/Data/shellagent-sft 640 \
    --dataset_mixer_list_config_names skill_tax_20260505_2.2k_combined_balanced_thinking_all \
    --gradient_checkpointing \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --seed 42
