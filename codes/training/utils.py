"""
Training utilities for TransMLLM

Helper functions for training setup, checkpointing, and monitoring.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import yaml


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to config YAML
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_training_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save training configuration to output directory.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "training_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculate model size statistics.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    Print model information.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    size_info = get_model_size(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {size_info['total_parameters']:,}")
    print(f"  Trainable parameters: {size_info['trainable_parameters']:,}")
    print(f"  Non-trainable parameters: {size_info['non_trainable_parameters']:,}")
    
    if size_info['total_parameters'] > 0:
        trainable_ratio = size_info['trainable_parameters'] / size_info['total_parameters'] * 100
        print(f"  Trainable ratio: {trainable_ratio:.2f}%")

