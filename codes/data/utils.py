"""
Data utilities for TransMLLM

Helper functions for data processing, validation, and formatting.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import json


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """
    Validate that image path exists and is readable.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return False
        
        # Try to open and verify it's a valid image
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
    
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        file_path: Path to save file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_image_paths(
    images_dir: Union[str, Path],
    image_ids: List[str],
    extensions: List[str] = [".jpg", ".png", ".jpeg"]
) -> Dict[str, Optional[Path]]:
    """
    Find image paths for given image IDs.
    
    Args:
        images_dir: Directory containing images
        image_ids: List of image IDs
        extensions: List of file extensions to try
    
    Returns:
        Dictionary mapping image_id to path (or None if not found)
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return {img_id: None for img_id in image_ids}
    
    result = {}
    for img_id in image_ids:
        found = False
        for ext in extensions:
            test_path = images_dir / f"{img_id}{ext}"
            if test_path.exists():
                result[img_id] = test_path
                found = True
                break
        
        if not found:
            result[img_id] = None
    
    return result


def format_translation_prompt(
    source_text: str,
    source_lang: str,
    target_lang: str
) -> str:
    """
    Format translation prompt.
    
    Args:
        source_text: Source text
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        Formatted prompt
    """
    return f"Translate from {source_lang} to {target_lang}: {source_text}"

