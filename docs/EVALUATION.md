# Evaluation Guide

## Overview

TransMLLM supports evaluation using multiple metrics:
- **BLEU**: N-gram overlap
- **COMET**: Neural semantic similarity
- **ROUGE**: Recall-oriented metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore**: Contextual embedding similarity

## Basic Evaluation

### Evaluate on PATIMT-Bench

```bash
bash scripts/evaluate.sh
```

Or directly:
```bash
python codes/evaluation/evaluate.py \
    --model_path ./checkpoints/transmllm_lora \
    --test_data_path ./data/patimt_multilingual/test \
    --output_dir ./evaluation_results \
    --metrics bleu comet rouge bertscore
```

### Evaluate on FLOES200

```bash
bash scripts/evaluate_flores200.sh
```

## Benchmark Evaluation

### PATIMT-Bench Evaluation

```bash
bash scripts/evaluate_patimt.sh
```

### FLOES200 Evaluation

```bash
bash scripts/evaluate_flores200.sh
```

## Understanding Results

Evaluation results are saved in the output directory with:
- Per-language-pair metrics
- Overall average metrics
- Detailed per-sample results (optional)

## Custom Evaluation

You can create custom evaluation scripts by importing the evaluation module:

```python
from codes.evaluation.metrics import compute_bleu, compute_comet, compute_rouge, compute_bertscore

# Compute metrics
bleu_score = compute_bleu(predictions, references)
comet_score = compute_comet(predictions, references, sources)
rouge_scores = compute_rouge(predictions, references)
bertscore_scores = compute_bertscore(predictions, references)
```

