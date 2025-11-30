"""
Simple inference example for TransMLLM

This script demonstrates how to use TransMLLM for document image translation.
"""

from codes.inference import inference

def main():
    # Example 1: Translate multilingual document (English to Kazakh)
    result = inference(
        model_path="./checkpoints/transmllm_lora",
        image_path="./examples/sample_data/multilingual-document/00.jpg",
        source_lang="en",
        target_lang="kk",
        output_path="./outputs/translation_en_kk.txt"
    )
    print(f"Translation result: {result}")
    
    # Example 2: Translate Kazakh document (Kazakh to English)
    result = inference(
        model_path="./checkpoints/transmllm_lora",
        image_path="./examples/sample_data/multilingual-multimodal-translation/kk/kk_0001.png",
        source_lang="kk",
        target_lang="en",
        output_path="./outputs/translation_kk_en.txt"
    )
    print(f"Translation result: {result}")
    
    # Example 3: Translate Uzbek document (Uzbek to English)
    result = inference(
        model_path="./checkpoints/transmllm_lora",
        image_path="./examples/sample_data/multilingual-multimodal-translation/uz/uz_0001.png",
        source_lang="uz",
        target_lang="en",
        output_path="./outputs/translation_uz_en.txt"
    )
    print(f"Translation result: {result}")
    
    # Example 4: Translate Urdu document (Urdu to English)
    result = inference(
        model_path="./checkpoints/transmllm_lora",
        image_path="./examples/sample_data/multilingual-multimodal-translation/ur/ur_0001.png",
        source_lang="ur",
        target_lang="en",
        output_path="./outputs/translation_ur_en.txt"
    )
    print(f"Translation result: {result}")

if __name__ == "__main__":
    main()

