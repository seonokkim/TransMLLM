"""
Simple training example for TransMLLM

This script demonstrates how to train TransMLLM using LoRA fine-tuning.
"""

from codes.training.train_lora import train_lora
import yaml

def main():
    # Load configuration
    with open("configs/train_lora.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Train model
    train_lora(config)

if __name__ == "__main__":
    main()

