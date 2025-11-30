#!/bin/bash

# TransMLLM LoRA Training Script

export CUDA_VISIBLE_DEVICES=0,1

python codes/training/train_lora.py \
    --config configs/train_lora.yaml \
    --output_dir ./checkpoints/transmllm_lora \
    --report_to tensorboard \
    --run_name transmllm_lora_train

