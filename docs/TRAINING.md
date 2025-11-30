# Training Guide

## Overview

TransMLLM supports two training strategies:
1. **LoRA Fine-tuning**: Parameter-efficient fine-tuning (recommended)
2. **Full Fine-tuning**: Full model fine-tuning

## LoRA Fine-tuning

### Configuration

Edit `configs/train_lora.yaml` to adjust training parameters:

```yaml
model:
  base_model: "llava-hf/llava-v1.6-mistral-7b-hf"
  lora_rank: 16
  lora_alpha: 32
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  # ... other parameters
```

### Training Command

```bash
bash scripts/train_lora.sh
```

Or directly:
```bash
python codes/training/train_lora.py \
    --config configs/train_lora.yaml \
    --output_dir ./checkpoints/transmllm_lora
```

## Full Fine-tuning

### Configuration

Edit `configs/train_full.yaml` to adjust training parameters.

### Training Command

```bash
bash scripts/train_full.sh
```

Or directly:
```bash
python codes/training/train_full.py \
    --config configs/train_full.yaml \
    --output_dir ./checkpoints/transmllm_full
```

## Training Tips

1. **Start with LoRA**: LoRA is more memory-efficient and faster to train
2. **Monitor Training**: Use TensorBoard to monitor training progress
3. **Save Checkpoints**: Checkpoints are saved automatically based on `save_steps`
4. **Resume Training**: Training can be resumed from checkpoints

## Hyperparameter Tuning

Key hyperparameters to tune:
- `learning_rate`: Start with 1e-5, adjust based on loss
- `lora_rank`: Higher rank = more parameters, better capacity
- `batch_size`: Adjust based on GPU memory
- `gradient_accumulation_steps`: Effective batch size = batch_size × gradient_accumulation_steps

