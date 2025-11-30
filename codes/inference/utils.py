"""
Inference utilities for TransMLLM

Helper functions for inference, batch processing, and output formatting.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
from PIL import Image


def load_images_from_paths(image_paths: List[Union[str, Path]]) -> List[Image.Image]:
    """
    Load multiple images from paths.
    
    Args:
        image_paths: List of image paths
    
    Returns:
        List of PIL Images
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            images.append(None)
    
    return images


def save_translation_results(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = "jsonl"
) -> None:
    """
    Save translation results to file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save file
        format: Output format ("jsonl" or "json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def format_translation_output(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translated_text: str,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format translation output as dictionary.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        source_text: Source text
        translated_text: Translated text
        image_path: Path to input image (optional)
    
    Returns:
        Formatted output dictionary
    """
    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_text": source_text,
        "translated_text": translated_text,
        "image_path": image_path,
    }

