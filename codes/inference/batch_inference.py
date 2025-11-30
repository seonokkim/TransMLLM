"""
Batch inference for TransMLLM
"""

import argparse
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm

from codes.inference.inference import inference


def batch_inference(
    model_path: str,
    input_file: str,
    output_file: str,
    source_lang: str = "en",
    target_lang: str = "ja",
    **kwargs
):
    """
    Run batch inference on multiple images
    
    Args:
        model_path: Path to trained model
        input_file: Path to input JSONL file with image paths
        output_file: Path to output JSONL file
        source_lang: Source language code
        target_lang: Target language code
        **kwargs: Additional inference arguments
    """
    # Load input data
    inputs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))
    
    # Run inference
    results = []
    for item in tqdm(inputs, desc="Processing"):
        image_path = item.get("image_path") or item.get("image")
        result = inference(
            model_path=model_path,
            image_path=image_path,
            source_lang=item.get("source_lang", source_lang),
            target_lang=item.get("target_lang", target_lang),
            **kwargs
        )
        
        results.append({
            "id": item.get("id", ""),
            "image": image_path,
            "source_lang": item.get("source_lang", source_lang),
            "target_lang": item.get("target_lang", target_lang),
            "translation": result,
        })
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language")
    parser.add_argument("--target_lang", type=str, default="ja", help="Target language")
    
    args = parser.parse_args()
    
    batch_inference(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )


if __name__ == "__main__":
    main()

