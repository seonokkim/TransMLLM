#!/bin/bash

# TransMLLM PATIMT-Bench Evaluation Script

python codes/evaluation/benchmark.py \
    --model_path ./checkpoints/transmllm_lora \
    --benchmark patimt \
    --test_data_path ./data/patimt_multilingual/test \
    --output_dir ./evaluation_results/patimt

