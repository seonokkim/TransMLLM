"""
Inference script for TransMLLM

Supports single image translation with multilingual support.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image

from codes.model.transmllm import TransMLLMModel
from codes.data.processor import TransMLLMProcessor
from codes.utils.device import get_device


def inference(
    model_path: str,
    image_path: str,
    source_lang: str,
    target_lang: str,
    output_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    **kwargs
) -> str:
    """
    Run inference on a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        source_lang: Source language code
        target_lang: Target language code
        output_path: Path to save output (optional)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        **kwargs: Additional generation parameters
    
    Returns:
        Translated text
    """
    # Validate inputs
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = TransMLLMModel(model_name=str(model_path))
    model.model.eval()
    
    # Load processor
    processor = TransMLLMProcessor(model_name=str(model_path))
    
    # Load and validate image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Build messages
    messages = processor.build_messages(
        image=image,
        source_text="",  # Will be set in prompt
        source_lang=source_lang,
        target_lang=target_lang,
    )
    
    # Update user message with actual source text if provided
    if "source_text" in kwargs:
        user_content = messages[0]["content"]
        for item in user_content:
            if item.get("type") == "text":
                item["text"] = f"Translate from {source_lang} to {target_lang}: {kwargs['source_text']}"
    
    # Process inputs
    try:
        inputs = processor(
            images=[image],
            messages=messages,
            return_tensors="pt",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to process inputs: {e}")
    
    # Move to device
    device = get_device()
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    print(f"Generating translation ({source_lang} -> {target_lang})...")
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **{k: v for k, v in kwargs.items() if k not in ["source_text"]}
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    # Decode
    if processor.tokenizer:
        generated_text = processor.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]
    else:
        generated_text = processor.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]
    
    # Extract translation (remove prompt)
    prompt = f"Translate from {source_lang} to {target_lang}:"
    if prompt in generated_text:
        translation = generated_text.split(prompt, 1)[-1].strip()
    else:
        translation = generated_text.strip()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        print(f"Translation saved to {output_path}")
    
    return translation


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TransMLLM Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to image"
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
        required=True,
        help="Target language"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling"
    )
    
    args = parser.parse_args()
    
    result = inference(
        model_path=args.model_path,
        image_path=args.image_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )
    
    print(f"\nTranslation ({args.source_lang} -> {args.target_lang}):")
    print(result)


if __name__ == "__main__":
    main()
