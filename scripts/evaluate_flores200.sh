#!/bin/bash

# TransMLLM FLOES200 Evaluation Script

python codes/evaluation/benchmark.py \
    --model_path ./checkpoints/transmllm_lora \
    --benchmark floes200 \
    --test_data_path ./data/floes200 \
    --output_dir ./evaluation_results/floes200

