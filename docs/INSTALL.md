# Installation Guide

## System Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (for full fine-tuning)
- 8GB+ GPU memory (for LoRA fine-tuning)
- 50GB+ disk space (for models and datasets)

## Step-by-Step Installation

### 1. Navigate to the Project Directory

```bash
cd TransMLLM
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n transmllm python=3.10
conda activate transmllm
```

### 3. Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Models

The base model will be automatically downloaded when you run training or inference. Alternatively, you can pre-download it:

```bash
bash scripts/download_models.sh
```

### 6. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:
1. Verify CUDA installation: `nvidia-smi`
2. Ensure PyTorch CUDA version matches your system CUDA version
3. Reinstall PyTorch with the correct CUDA version

### Memory Issues

If you run out of GPU memory:
1. Reduce batch size in config files
2. Use gradient checkpointing (already enabled by default)
3. Use LoRA instead of full fine-tuning
4. Use QLoRA (4-bit quantization) for even lower memory usage

### Dataset Download Issues

If dataset download fails:
1. Check internet connection
2. Verify HuggingFace authentication: `huggingface-cli login`
3. Manually download datasets from HuggingFace

### Dataset Cache Location

HuggingFace datasets are automatically cached to:
- **Windows**: `C:\Users\<username>\.cache\huggingface\datasets\`
- **Linux/Mac**: `~/.cache/huggingface/datasets/`

The datasets will be automatically downloaded on first use. Images are embedded in the dataset, so no separate image directory is needed when using HuggingFace datasets.

To clear cache:
```bash
# Remove specific dataset
rm -r ~/.cache/huggingface/datasets/rileykim___multilingual-document

# Or set custom cache directory
export HF_HOME=/path/to/custom/cache
```

