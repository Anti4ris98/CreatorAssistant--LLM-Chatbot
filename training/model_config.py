"""Model configuration utilities for CreatorAssistant training."""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Configuration for base model settings."""
    name: str = "meta-llama/Llama-2-7b-chat-hf"
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    max_seq_length: int = 512
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./models/finetuned"
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
