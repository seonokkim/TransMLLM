# Frequently Asked Questions (FAQ)

## General Questions

### What is TransMLLM?

TransMLLM is a multimodal multilingual document translation model that uses Large Multimodal Models (LMMs) to translate document images across 11 languages.

### Which languages are supported?

TransMLLM supports 11 languages:
- English (en)
- Vietnamese (vi)
- Indonesian (id)
- Uzbek (uz)
- Russian (ru)
- Japanese (ja)
- Kazakh (kk)
- Chinese Simplified (zh-cn)
- Chinese Traditional (zh-tw)
- Korean (ko)
- Urdu (ur)

## Training Questions

### What's the difference between LoRA and Full Fine-tuning?

- **LoRA**: Parameter-efficient fine-tuning, trains only ~0.5M parameters (0.007% of model)
- **Full Fine-tuning**: Trains all 7B parameters

LoRA is recommended for most use cases as it's faster and requires less memory.

### How much GPU memory do I need?

- **LoRA**: 8GB+ GPU memory
- **Full Fine-tuning**: 16GB+ GPU memory

### How long does training take?

Training time depends on:
- Dataset size
- Number of epochs
- GPU specifications

For the full dataset (10,600 samples), LoRA training typically takes 2-4 hours on a single A100 GPU.

## Inference Questions

### How do I use the model for inference?

```bash
# Example: Translate multilingual document (English to Kazakh)
python codes/inference/inference.py \
    --model_path ./checkpoints/transmllm_lora \
    --image_path ./examples/sample_data/multilingual-document/00.jpg \
    --source_lang en \
    --target_lang kk \
    --output_path ./outputs/translation.txt

# Example: Translate Kazakh document (Kazakh to English)
python codes/inference/inference.py \
    --model_path ./checkpoints/transmllm_lora \
    --image_path ./examples/sample_data/multilingual-multimodal-translation/kk/kk_0001.png \
    --source_lang kk \
    --target_lang en \
    --output_path ./outputs/translation_kk_en.txt
```

### What image formats are supported?

The model supports common image formats: JPEG, PNG, etc. Images are automatically preprocessed.

## Dataset Questions

### Where can I download the datasets?

Datasets are available on HuggingFace:
- Primary dataset: [rileykim/multilingual-document](https://huggingface.co/datasets/rileykim/multilingual-document)
- FLORES200-based: [rileykim/multilingual-image-text-translation](https://huggingface.co/datasets/rileykim/multilingual-image-text-translation)

### How do I use custom datasets?

You can create custom datasets following the same format as the HuggingFace datasets. See the dataset documentation for details.

## Technical Questions

### What base model does TransMLLM use?

TransMLLM is based on LLaVA-Next-7B (llava-hf/llava-v1.6-mistral-7b-hf), which uses:
- Vision Encoder: CLIP ViT-L/14
- Language Model: Mistral-7B

### Can I use this model commercially?

Yes, the model is released under the MIT License, which allows commercial use.

## Troubleshooting

### I'm getting CUDA out of memory errors

Try:
1. Reduce batch size
2. Use gradient checkpointing (already enabled)
3. Use LoRA instead of full fine-tuning
4. Use a smaller model or QLoRA

### Training is very slow

Try:
1. Use multiple GPUs with distributed training
2. Increase batch size (if memory allows)
3. Use mixed precision training (bf16, already enabled)

### Model predictions are poor

Try:
1. Train for more epochs
2. Adjust learning rate
3. Use more training data
4. Try full fine-tuning instead of LoRA

