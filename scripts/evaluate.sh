#!/bin/bash

# TransMLLM Evaluation Script

python codes/evaluation/evaluate.py \
    --model_path ./checkpoints/transmllm_lora \
    --test_data_path ./data/patimt_multilingual/test \
    --output_dir ./evaluation_results \
    --metrics bleu comet rouge bertscore

