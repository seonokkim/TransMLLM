"""
LoRA Adapter for TransMLLM

Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Optional


def setup_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    use_4bit: bool = False,
    **kwargs
):
    """
    Setup LoRA adapter for the model
    
    Args:
        model: Base model to apply LoRA
        rank: LoRA rank (r)
        alpha: LoRA alpha (α)
        target_modules: List of module names to apply LoRA
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        **kwargs: Additional LoRA config parameters
    
    Returns:
        Model with LoRA adapter
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=kwargs.get("lora_dropout", 0.05),
        bias=kwargs.get("bias", "none"),
        task_type=kwargs.get("task_type", "CAUSAL_LM"),
    )
    
    model = get_peft_model(model, lora_config)
    return model

