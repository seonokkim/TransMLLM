#!/bin/bash

# TransMLLM Inference Script

# Example 1: Translate multilingual document (English to Kazakh)
python codes/inference/inference.py \
    --model_path ./checkpoints/transmllm_lora \
    --image_path ./examples/sample_data/multilingual-document/00.jpg \
    --source_lang en \
    --target_lang kk \
    --output_path ./outputs/translation_en_kk.txt

# Example 2: Translate Kazakh document (Kazakh to English)
python codes/inference/inference.py \
    --model_path ./checkpoints/transmllm_lora \
    --image_path ./examples/sample_data/multilingual-multimodal-translation/kk/kk_0001.png \
    --source_lang kk \
    --target_lang en \
    --output_path ./outputs/translation_kk_en.txt

# Example 3: Translate Uzbek document (Uzbek to English)
# python codes/inference/inference.py \
#     --model_path ./checkpoints/transmllm_lora \
#     --image_path ./examples/sample_data/multilingual-multimodal-translation/uz/uz_0001.png \
#     --source_lang uz \
#     --target_lang en \
#     --output_path ./outputs/translation_uz_en.txt

# Example 4: Translate Urdu document (Urdu to English)
# python codes/inference/inference.py \
#     --model_path ./checkpoints/transmllm_lora \
#     --image_path ./examples/sample_data/multilingual-multimodal-translation/ur/ur_0001.png \
#     --source_lang ur \
#     --target_lang en \
#     --output_path ./outputs/translation_ur_en.txt

