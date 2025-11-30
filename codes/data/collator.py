"""
Data collator for batching TransMLLM training samples.

Handles multimodal inputs with proper label masking for training.
Supports both HuggingFace datasets (with embedded PIL Images) and local files.
"""

from typing import Dict, List, Any, Optional
import torch
from PIL import Image

from codes.data.processor import TransMLLMProcessor
from codes.data.dataset import SampleItem


class TransMLLMCollator:
    """
    Data collator for TransMLLM training.
    
    Processes batches of SampleItem objects into model inputs with proper
    label masking (prompt tokens masked, only target text used for loss).
    
    Supports:
    - HuggingFace datasets: Images are PIL Image objects in SampleItem.image
    - Local files: Images are loaded from SampleItem.image_path
    """
    
    def __init__(
        self,
        processor: TransMLLMProcessor,
        max_length: int = 2048,
        image_required: bool = True,
    ) -> None:
        """
        Initialize collator.
        
        Args:
            processor: TransMLLMProcessor instance
            max_length: Maximum sequence length
            image_required: Whether images are required
        """
        self.processor = processor
        self.max_length = max_length
        self.image_required = image_required
        self.tokenizer = processor.tokenizer
    
    def __call__(self, batch: List[SampleItem]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            batch: List of SampleItem objects
        
        Returns:
            Batched features with:
            - input_ids: Tokenized input
            - attention_mask: Attention mask
            - pixel_values: Processed images
            - labels: Labels with prompt tokens masked
        """
        images: List[Optional[Image.Image]] = []
        
        # Load images
        for item in batch:
            if item.image is not None:
                # HuggingFace dataset: image is already a PIL Image
                images.append(item.image)
            elif item.image_path is not None:
                # Local file: load from path
                try:
                    img = Image.open(item.image_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    if self.image_required:
                        raise ValueError(f"Failed to load image {item.image_path}: {e}")
                    images.append(None)
            else:
                if self.image_required:
                    raise ValueError("No image found in sample item")
                images.append(None)
        
        # Process each sample
        all_input_ids: List[torch.Tensor] = []
        all_attention_masks: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pixel_values: List[torch.Tensor] = []
        all_image_sizes: List[torch.Tensor] = []
        
        for item, img in zip(batch, images):
            # Build messages
            messages = self.processor.build_messages(
                image=img if self.image_required else None,
                source_text=item.source_text,
                target_text=item.target_text,
                source_lang=item.source_lang,
                target_lang=item.target_lang,
            )
            
            # Process with processor
            try:
                encoded = self.processor(
                    images=[img] if img is not None else None,
                    messages=messages,
                    return_tensors="pt",
                    truncation=False,
                )
            except Exception as e:
                # Fallback: process without image if image processing fails
                if img is not None:
                    print(f"Warning: Image processing failed, using text only: {e}")
                    encoded = self.processor(
                        images=None,
                        messages=messages,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                else:
                    raise
            
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))[0]
            
            # Create labels: mask prompt tokens, keep only target text
            labels = self._create_labels(input_ids, item.target_text)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
            
            # Extract pixel values and image sizes
            if img is not None and "pixel_values" in encoded:
                pixel_values = encoded["pixel_values"]
                # Handle different shapes: [batch, ...] or [...]
                if len(pixel_values.shape) >= 4:
                    all_pixel_values.append(pixel_values[0] if len(pixel_values.shape) > 3 else pixel_values)
                else:
                    all_pixel_values.append(pixel_values)
                
                # Extract image sizes
                if "image_sizes" in encoded:
                    img_sizes = encoded["image_sizes"]
                    if len(img_sizes.shape) >= 2:
                        all_image_sizes.append(img_sizes[0])
                    else:
                        all_image_sizes.append(img_sizes)
                else:
                    # Create from image dimensions
                    img_size = torch.tensor([img.height, img.width], dtype=torch.long)
                    all_image_sizes.append(img_size)
        
        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for ids, attn, labels in zip(all_input_ids, all_attention_masks, all_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
            padded_attention_masks.append(torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)]))
        
        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
        }
        
        # Add pixel values if available
        if all_pixel_values:
            # Stack pixel values (handle variable shapes)
            try:
                result["pixel_values"] = torch.stack(all_pixel_values)
            except Exception:
                # If shapes don't match, use list (model should handle this)
                result["pixel_values"] = all_pixel_values
        
        # Add image sizes if available
        if all_image_sizes:
            try:
                result["image_sizes"] = torch.stack(all_image_sizes)
            except Exception:
                result["image_sizes"] = all_image_sizes
        
        return result
    
    def _create_labels(
        self, 
        input_ids: torch.Tensor, 
        target_text: str
    ) -> torch.Tensor:
        """
        Create labels with prompt tokens masked.
        
        Args:
            input_ids: Full input token IDs
            target_text: Target translation text
        
        Returns:
            Labels tensor with prompt tokens set to -100
        """
        if self.tokenizer is None:
            # Simple approach: mask first half (approximate)
            labels = input_ids.clone()
            labels[:len(labels) // 2] = -100
            return labels
        
        # Tokenize target text
        target_ids = self.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"][0]
        
        # Create labels: mask everything except target tokens
        labels = input_ids.clone()
        labels[:] = -100
        
        # Find target tokens at the end and unmask them
        target_len = min(len(target_ids), len(labels))
        labels[-target_len:] = input_ids[-target_len:]
        
        return labels
