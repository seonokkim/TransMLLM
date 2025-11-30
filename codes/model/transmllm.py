"""
TransMLLM Model Architecture

Based on LLaVA-Next-7B (llava-hf/llava-v1.6-mistral-7b-hf) for multilingual document translation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List

from transformers import AutoModelForCausalLM, AutoProcessor


class TransMLLMModel(nn.Module):
    """
    TransMLLM Model for Multilingual Document Translation.
    
    Architecture:
    - Vision Encoder: CLIP ViT-L/14 (from LLaVA-Next)
    - Language Model: Mistral-7B
    - Multimodal Projector: Vision-to-language projection
    
    Supports both training and inference modes.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict] = "auto",
        **kwargs
    ) -> None:
        """
        Initialize TransMLLM model.
        
        Args:
            model_name: Model name or path
            torch_dtype: Data type for model (default: bfloat16)
            device_map: Device mapping strategy
            **kwargs: Additional model loading arguments
        """
        super().__init__()
        self.model_name = model_name
        
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        Forward pass for training.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask
            pixel_values: Preprocessed images
            image_sizes: Image sizes [height, width] (for LLaVA Next)
            labels: Target labels for training
            **kwargs: Additional model arguments
        
        Returns:
            Model outputs with loss (if labels provided)
        """
        # Prepare inputs
        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
        }
        
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        
        if labels is not None:
            model_inputs["labels"] = labels
        
        return self.model(**model_inputs, **kwargs)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate translation.
        
        Args:
            input_ids: Tokenized input text
            pixel_values: Preprocessed images
            image_sizes: Image sizes [height, width] (for LLaVA Next)
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Generated token ids
        """
        # Prepare generation inputs
        generation_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        
        if pixel_values is not None:
            generation_inputs["pixel_values"] = pixel_values
        
        if image_sizes is not None:
            generation_inputs["image_sizes"] = image_sizes
        
        if attention_mask is not None:
            generation_inputs["attention_mask"] = attention_mask
        
        return self.model.generate(**generation_inputs, **generation_kwargs)
    
    def eval(self) -> "TransMLLMModel":
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True) -> "TransMLLMModel":
        """Set model to training mode"""
        self.model.train(mode)
        return self
