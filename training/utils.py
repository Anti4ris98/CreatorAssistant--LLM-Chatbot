"""Utility functions for training pipeline."""

import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def print_model_info(model):
    """Print model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print("="*50 + "\n")


def create_directories(config):
    """Create all necessary directories for training."""
    directories = [
        config['training']['output_dir'],
        "logs",
        "data/processed",
        "data/raw"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def estimate_training_time(num_samples, batch_size, num_epochs, time_per_batch=2.0):
    """Estimate total training time."""
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    estimated_seconds = total_steps * time_per_batch
    
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    
    print(f"\n⏱️  Estimated training time: {hours}h {minutes}m")
    print(f"   Total steps: {total_steps}")
    print(f"   Steps per epoch: {steps_per_epoch}\n")
