"""
Custom Trainer class for TransMLLM

Extends HuggingFace Trainer with custom functionality for multimodal training.
"""

from typing import Optional, Dict, Any, List
import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from codes.model.transmllm import TransMLLMModel


class TransMLLMTrainer(Trainer):
    """
    Custom trainer for TransMLLM with multimodal support.
    
    Extends HuggingFace Trainer to handle:
    - Multimodal inputs (images + text)
    - Custom loss computation
    - Image size handling for LLaVA Next
    """
    
    def __init__(
        self,
        model: TransMLLMModel,
        args: TrainingArguments,
        **kwargs
    ):
        """
        Initialize custom trainer.
        
        Args:
            model: TransMLLM model
            args: Training arguments
            **kwargs: Additional trainer arguments
        """
        super().__init__(model=model.model, args=args, **kwargs)
        self.transmllm_model = model
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ):
        """
        Compute training loss.
        
        Args:
            model: Model to compute loss for
            inputs: Input tensors
            return_outputs: Whether to return model outputs
        
        Returns:
            Loss value (and outputs if return_outputs=True)
        """
        # Extract inputs
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        
        loss = outputs.loss if hasattr(outputs, "loss") else None
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step.
        
        Args:
            model: Model to use
            inputs: Input tensors
            prediction_loss_only: Whether to compute loss only
            ignore_keys: Keys to ignore
        
        Returns:
            Tuple of (loss, logits, labels)
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
                outputs = model(**inputs)
            
            if isinstance(outputs, dict):
                logits = outputs.get("logits")
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else None
        
        if prediction_loss_only:
            return (loss, None, None)
        
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        
        return (loss, logits, labels)

