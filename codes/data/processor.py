"""
Data preprocessing and tokenization for TransMLLM.

Handles multimodal chat template formatting for LLaVA-Next models.
"""

from typing import Dict, List, Optional, Union, Any
from PIL import Image

from transformers import AutoProcessor


class TransMLLMProcessor:
    """
    Processor for TransMLLM that handles image and text preprocessing.
    
    Supports LLaVA-Next chat template format with multimodal content.
    """
    
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf") -> None:
        """
        Initialize processor.
        
        Args:
            model_name: Model name or path
        """
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
    
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        text: Optional[Union[str, List[str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process images and text.
        
        Args:
            images: Single image or list of images
            text: Single text or list of texts (alternative to messages)
            messages: Chat messages in LLaVA format (preferred)
            return_tensors: Return format ("pt", "np", etc.)
            **kwargs: Additional processor arguments
        
        Returns:
            Processed inputs with pixel_values, input_ids, attention_mask, etc.
        """
        # Handle LLaVA Next format with messages
        if messages is not None:
            # Check if processor supports messages directly
            processor_name = type(self.processor).__name__.lower()
            is_llava_next = "llavanext" in processor_name or "llava_next" in processor_name
            
            if is_llava_next and images:
                try:
                    # LLaVA Next can process messages with images directly
                    return self.processor(
                        messages,
                        images=images if isinstance(images, list) else [images],
                        return_tensors=return_tensors,
                        padding="longest",
                        truncation=False,
                        **kwargs
                    )
                except Exception:
                    # Fallback to text-based processing
                    pass
            
            # Fallback: convert messages to text
            if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
        
        # Standard processing
        return self.processor(
            images=images,
            text=text,
            return_tensors=return_tensors,
            padding="longest" if return_tensors == "pt" else False,
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", 2048),
            **{k: v for k, v in kwargs.items() if k not in ["truncation", "max_length"]}
        )
    
    def build_messages(
        self,
        image: Optional[Image.Image],
        source_text: str,
        target_text: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "ja",
    ) -> List[Dict[str, Any]]:
        """
        Build chat messages in LLaVA format.
        
        Args:
            image: Input image (optional)
            source_text: Source text to translate
            target_text: Target translation (for training)
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            List of message dictionaries
        """
        user_content: List[Dict[str, Any]] = []
        
        if image is not None:
            user_content.append({"type": "image", "image": image})
        
        prompt = f"Translate from {source_lang} to {target_lang}: {source_text}"
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        if target_text is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}]
            })
        
        return messages
