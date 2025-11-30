#!/bin/bash

# TransMLLM Full Fine-tuning Script

export CUDA_VISIBLE_DEVICES=0,1

python codes/training/train_full.py \
    --config configs/train_full.yaml \
    --output_dir ./checkpoints/transmllm_full \
    --report_to tensorboard \
    --run_name transmllm_full_train

