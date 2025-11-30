"""
Device management utilities
"""

import torch


def get_device() -> torch.device:
    """
    Get available device (CUDA if available, else CPU)
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_device(device_id: int = 0):
    """
    Setup CUDA device
    
    Args:
        device_id: CUDA device ID
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        print("CUDA not available, using CPU")

