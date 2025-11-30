"""
Evaluation script for TransMLLM

Evaluates model performance using multiple metrics (BLEU, COMET, ROUGE, BERTScore).
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from codes.inference.inference import inference
from codes.evaluation.metrics import (
    compute_bleu,
    compute_comet,
    compute_rouge,
    compute_bertscore
)
from codes.utils.logging import setup_logging


def evaluate(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    metrics: List[str] = ["bleu", "comet", "rouge", "bertscore"],
    images_dir: Optional[str] = None,
    source_lang: str = "en",
    target_lang: str = "ja",
    max_samples: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data (JSONL file)
        output_dir: Output directory for results
        metrics: List of metrics to compute
        images_dir: Directory containing images
        source_lang: Source language code
        target_lang: Target language code
        max_samples: Maximum number of samples to evaluate
        **kwargs: Additional inference arguments
    
    Returns:
        Dictionary with evaluation results
    """
    # Setup logging
    setup_logging(log_dir=str(Path(output_dir) / "logs"))
    
    # Validate inputs
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    test_data_path = Path(test_data_path)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data path not found: {test_data_path}")
    
    # Load test data
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if line.strip():
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line {idx}: {e}")
                    continue
    
    if not test_data:
        raise ValueError("No valid test data found")
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    # Run inference
    predictions: List[str] = []
    references: List[str] = []
    sources: List[str] = []
    image_paths: List[str] = []
    
    for item in tqdm(test_data, desc="Running inference"):
        # Get image path
        image_path = item.get("image_path") or item.get("image")
        if images_dir:
            image_id = item.get("image_id") or item.get("id", "")
            image_path = Path(images_dir) / f"{image_id}.jpg"
            if not image_path.exists():
                image_path = Path(images_dir) / f"{image_id}.png"
        
        if not image_path or not Path(image_path).exists():
            print(f"Warning: Image not found for item {item.get('id', 'unknown')}, skipping")
            continue
        
        # Get source and target text
        source_text = item.get("text") or item.get("source_text", "")
        target_text = item.get("target_text") or item.get("reference", "")
        
        if not source_text or not target_text:
            print(f"Warning: Missing text for item {item.get('id', 'unknown')}, skipping")
            continue
        
        # Get language codes
        item_source_lang = item.get("source_lang", source_lang)
        item_target_lang = item.get("target_lang", target_lang)
        
        # Run inference
        try:
            pred = inference(
                model_path=str(model_path),
                image_path=str(image_path),
                source_lang=item_source_lang,
                target_lang=item_target_lang,
                source_text=source_text,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                **{k: v for k, v in kwargs.items() if k not in ["max_new_tokens", "temperature"]}
            )
            
            predictions.append(pred)
            references.append(target_text)
            sources.append(source_text)
            image_paths.append(str(image_path))
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue
    
    if not predictions:
        raise ValueError("No successful predictions")
    
    print(f"Successfully processed {len(predictions)} samples")
    
    # Compute metrics
    results: Dict[str, Any] = {
        "num_samples": len(predictions),
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    
    if "bleu" in metrics:
        try:
            results["bleu"] = compute_bleu(predictions, references)
            print(f"BLEU: {results['bleu']:.4f}")
        except Exception as e:
            print(f"BLEU computation failed: {e}")
            results["bleu"] = None
    
    if "comet" in metrics:
        try:
            comet_score = compute_comet(predictions, references, sources)
            if comet_score is not None:
                results["comet"] = comet_score
                print(f"COMET: {results['comet']:.4f}")
            else:
                results["comet"] = None
        except Exception as e:
            print(f"COMET computation failed: {e}")
            results["comet"] = None
    
    if "rouge" in metrics:
        try:
            rouge_scores = compute_rouge(predictions, references)
            results["rouge"] = rouge_scores
            print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        except Exception as e:
            print(f"ROUGE computation failed: {e}")
            results["rouge"] = None
    
    if "bertscore" in metrics:
        try:
            bertscore_scores = compute_bertscore(predictions, references, lang=target_lang)
            if bertscore_scores is not None:
                results["bertscore"] = bertscore_scores
                print(f"BERTScore F1: {bertscore_scores['f1']:.4f}")
            else:
                results["bertscore"] = None
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            results["bertscore"] = None
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save detailed predictions
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for pred, ref, src, img_path in zip(predictions, references, sources, image_paths):
            f.write(json.dumps({
                "prediction": pred,
                "reference": ref,
                "source": src,
                "image_path": img_path
            }, ensure_ascii=False) + "\n")
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")
    
    return results


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate TransMLLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test data (JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "comet", "rouge", "bertscore"],
        choices=["bleu", "comet", "rouge", "bertscore"],
        help="Metrics to compute"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Directory containing images"
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
        help="Source language"
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="ja",
        help="Target language"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    
    args = parser.parse_args()
    
    results = evaluate(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        metrics=args.metrics,
        images_dir=args.images_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    print("\n=== Final Results ===")
    for metric, score in results.items():
        if metric not in ["num_samples", "source_lang", "target_lang"]:
            if isinstance(score, dict):
                print(f"{metric}:")
                for k, v in score.items():
                    print(f"  {k}: {v:.4f}")
            elif score is not None:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()
