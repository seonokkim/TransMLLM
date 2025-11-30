"""
Dataset classes for TransMLLM training and evaluation

Supports HuggingFace datasets and local JSONL files with multilingual document images.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from PIL import Image


@dataclass
class SampleItem:
    """Single sample item for training"""
    image: Optional[Image.Image]  # PIL Image (for HuggingFace) or None
    image_path: Optional[str]  # Path string (for local files) or None
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str


class MultilingualDocumentDataset(TorchDataset):
    """
    Dataset for multilingual document translation.
    
    Supports:
    - HuggingFace datasets (rileykim/multilingual-document)
      * Images are embedded in the dataset (PIL Image objects)
      * Automatically cached to ~/.cache/huggingface/datasets/
    - Local JSONL files with image paths
      * Requires images_dir to be specified
    """
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
        images_dir: Optional[Union[str, Path]] = None,
        source_lang: str = "en",
        target_lang: str = "ja",
        processor: Optional[Any] = None,
        split: str = "train",
        limit: Optional[int] = None,
        image_ext: str = ".jpg",
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., "rileykim/multilingual-document")
            data_path: Local path to JSONL file
            images_dir: Directory containing images (required for local JSONL)
            source_lang: Source language code
            target_lang: Target language code
            processor: Image-text processor (optional, for compatibility)
            split: Dataset split (train/val/test)
            limit: Limit number of samples (for testing)
            image_ext: Image file extension (for local files)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.processor = processor
        self.images_dir = Path(images_dir) if images_dir else None
        self.image_ext = image_ext
        self.items: List[SampleItem] = []
        self.is_huggingface = False
        
        # Load data
        if dataset_name:
            self._load_huggingface_dataset(dataset_name, split, limit)
            self.is_huggingface = True
        elif data_path:
            self._load_jsonl(data_path, limit)
        else:
            raise ValueError("Either dataset_name or data_path must be provided")
    
    def _load_huggingface_dataset(
        self, 
        dataset_name: str, 
        split: str, 
        limit: Optional[int] = None
    ) -> None:
        """
        Load dataset from HuggingFace.
        
        Note: HuggingFace datasets are automatically cached to:
        - Windows: C:\Users\<username>\.cache\huggingface\datasets\
        - Linux/Mac: ~/.cache/huggingface/datasets/
        """
        try:
            dataset = load_dataset(dataset_name, split=split)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            for item in dataset:
                self._process_huggingface_item(item)
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset: {e}")
    
    def _load_jsonl(self, path: Union[str, Path], limit: Optional[int] = None) -> None:
        """Load JSONL file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        if not self.images_dir:
            raise ValueError("images_dir must be provided when using local JSONL files")
        
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line {idx}: {e}")
                        continue
        
        for item in data:
            self._process_jsonl_item(item)
    
    def _process_huggingface_item(self, item: Dict[str, Any]) -> None:
        """Process item from HuggingFace dataset"""
        # Extract image (should be PIL Image in HuggingFace datasets)
        image = None
        if "image" in item:
            if isinstance(item["image"], Image.Image):
                image = item["image"]
            elif isinstance(item["image"], dict) and "path" in item["image"]:
                # Image path in dict format
                image_path = item["image"]["path"]
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    pass
        
        # Extract texts
        source_text, target_text = self._extract_texts(item)
        
        if not source_text or not target_text:
            return  # Skip if text missing
        
        self.items.append(
            SampleItem(
                image=image,
                image_path=None,
                source_text=source_text,
                target_text=target_text,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
        )
    
    def _process_jsonl_item(self, item: Dict[str, Any]) -> None:
        """Process item from local JSONL file"""
        # Get image path
        image_id = item.get("image_id") or item.get("id") or item.get("image", "")
        
        if not image_id:
            return  # Skip if no image ID
        
        # Try to find image file
        image_path = None
        if self.images_dir:
            # Try language-specific directory first
            lang_dir = self.images_dir / self.source_lang
            test_path = lang_dir / f"{image_id}{self.image_ext}"
            if test_path.exists():
                image_path = str(test_path)
            else:
                # Fallback to root images directory
                test_path = self.images_dir / f"{image_id}{self.image_ext}"
                if test_path.exists():
                    image_path = str(test_path)
        
        # Try direct path
        if not image_path and isinstance(image_id, str):
            test_path = Path(image_id)
            if test_path.exists():
                image_path = str(test_path)
        
        if not image_path:
            return  # Skip if image not found
        
        # Extract texts
        source_text, target_text = self._extract_texts(item)
        
        if not source_text or not target_text:
            return  # Skip if text missing
        
        self.items.append(
            SampleItem(
                image=None,
                image_path=image_path,
                source_text=source_text,
                target_text=target_text,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
        )
    
    def _extract_texts(self, item: Dict[str, Any]) -> tuple[str, str]:
        """Extract source and target texts from item"""
        source_text = ""
        target_text = ""
        
        # Handle merge_ocr format (PATIMT-Multilingual)
        if "merge_ocr" in item:
            for ocr_item in item["merge_ocr"]:
                if ocr_item.get("src_lang") == "English" or ocr_item.get("src_lang") == self.source_lang:
                    source_text = ocr_item.get("src_text", "")
                if ocr_item.get("tgt_lang") and ocr_item.get("tgt_lang") != "English":
                    target_text = ocr_item.get("tgt_text", "")
        
        # Handle standard format
        if not source_text:
            source_text = item.get("text") or item.get("source_text") or ""
        if not target_text:
            target_text = item.get("target_text") or item.get("target") or ""
        
        return source_text, target_text
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> SampleItem:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            SampleItem with image (PIL Image) or image_path (string)
        """
        if idx >= len(self.items):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.items)}")
        
        return self.items[idx]
