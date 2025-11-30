"""
Vision Encoder Wrapper for TransMLLM

Wraps the CLIP vision encoder from LLaVA-Next for document image processing.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from transformers import AutoModel, AutoProcessor


class VisionEncoder(nn.Module):
    """
    Vision encoder wrapper for TransMLLM.
    
    Uses CLIP ViT-L/14 from LLaVA-Next for encoding document images.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        **kwargs
    ) -> None:
        """
        Initialize vision encoder.
        
        Args:
            model_name: Model name (vision encoder is part of the full model)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_name = model_name
        
        # Load processor to access vision encoder
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Note: In LLaVA-Next, the vision encoder is part of the full model
        # This wrapper provides a clean interface for vision-only operations
        self._vision_encoder = None
    
    def _get_vision_encoder(self, model):
        """
        Extract vision encoder from full model.
        
        Args:
            model: Full LLaVA model
        
        Returns:
            Vision encoder component
        """
        if self._vision_encoder is None:
            # LLaVA-Next models have vision_tower attribute
            if hasattr(model, "vision_tower"):
                self._vision_encoder = model.vision_tower
            elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
                self._vision_encoder = model.model.vision_tower
            else:
                raise AttributeError("Vision encoder not found in model")
        
        return self._vision_encoder
    
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode images to vision features.
        
        Args:
            pixel_values: Preprocessed image tensors
            image_sizes: Image sizes [height, width]
            **kwargs: Additional arguments
        
        Returns:
            Vision features
        """
        if self._vision_encoder is None:
            raise RuntimeError("Vision encoder not initialized. Call with model first.")
        
        return self._vision_encoder(pixel_values, image_sizes=image_sizes, **kwargs)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            pixel_values: Preprocessed image tensors
            image_sizes: Image sizes [height, width]
            **kwargs: Additional arguments
        
        Returns:
            Vision features
        """
        return self.encode_image(pixel_values, image_sizes=image_sizes, **kwargs)

