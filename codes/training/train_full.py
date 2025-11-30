"""
Full Fine-tuning Script for TransMLLM

Supports full parameter fine-tuning of the entire model.
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)

from codes.model.transmllm import TransMLLMModel
from codes.data.dataset import MultilingualDocumentDataset
from codes.data.collator import TransMLLMCollator
from codes.data.processor import TransMLLMProcessor
from codes.utils.logging import setup_logging


def train_full(
    config_path: str,
    output_dir: str,
    report_to: str = "tensorboard",
    run_name: str = "transmllm_full",
    **kwargs
) -> None:
    """
    Full fine-tuning of TransMLLM.
    
    Args:
        config_path: Path to training config YAML
        output_dir: Output directory for checkpoints
        report_to: Reporting tool (tensorboard, wandb, etc.)
        run_name: Run name for logging
        **kwargs: Additional training arguments
    """
    # Setup logging
    setup_logging(log_dir=str(Path(output_dir) / "logs"))
    
    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Load model (full fine-tuning, no LoRA)
    model = TransMLLMModel(
        model_name=config["model"]["base_model"]
    )
    
    # Load processor
    processor = TransMLLMProcessor(
        model_name=config["model"]["base_model"]
    )
    
    # Load datasets
    train_dataset = MultilingualDocumentDataset(
        dataset_name=config["data"].get("dataset_name"),
        data_path=config["data"].get("train_data_path"),
        images_dir=config["data"].get("images_dir"),
        source_lang=config.get("source_lang", "en"),
        target_lang=config.get("target_lang", "ja"),
        processor=processor,
        split="train",
        limit=config.get("data", {}).get("limit"),
    )
    
    val_dataset = None
    if config["data"].get("val_data_path"):
        val_dataset = MultilingualDocumentDataset(
            dataset_name=config["data"].get("dataset_name"),
            data_path=config["data"].get("val_data_path"),
            images_dir=config["data"].get("images_dir"),
            source_lang=config.get("source_lang", "en"),
            target_lang=config.get("target_lang", "ja"),
            processor=processor,
            split="val",
            limit=config.get("data", {}).get("val_limit"),
        )
    
    # Data collator
    collator = TransMLLMCollator(
        processor=processor,
        max_length=config["data"].get("max_length", 2048),
        image_required=True,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        eval_steps=config["training"].get("eval_steps", 50),
        save_total_limit=config["training"]["save_total_limit"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        report_to=report_to,
        run_name=run_name,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        **kwargs
    )
    
    # Callbacks
    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 3)
        ))
    
    # Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    
    # Train
    print(f"Starting full fine-tuning with {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation set: {len(val_dataset)} samples")
    
    trainer.train()
    
    # Save final model
    trainer.save_model()
    processor.processor.save_pretrained(output_dir)
    
    print(f"Training completed. Model saved to {output_dir}")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Full fine-tuning of TransMLLM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting tool"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="transmllm_full",
        help="Run name"
    )
    
    args = parser.parse_args()
    
    train_full(
        config_path=args.config,
        output_dir=args.output_dir,
        report_to=args.report_to,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
