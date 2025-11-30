#!/bin/bash

# TransMLLM Model Download Script

# Download base model (LLaVA-Next-7B)
# This will be automatically downloaded when running training/inference
# But you can pre-download it using:

python -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
    model = AutoModelForCausalLM.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf'); \
    processor = AutoProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"

echo "Base model downloaded successfully!"

