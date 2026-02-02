import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import yaml
from pathlib import Path

def load_config():
    with open("config/training_config.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_model_and_tokenizer(config):
    print("🔧 Loading model and tokenizer...")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_training_data(tokenizer, config):
    print("📚 Loading data...")
    
    dataset = load_dataset('json', data_files={
        'train': 'data/processed/train.json',
        'eval': 'data/processed/eval.json'
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['training']['max_seq_length'],
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset

def train():
    config = load_config()
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(config)
    dataset = load_training_data(tokenizer, config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval']
    )
    
    print("🚀 Starting training...")
    trainer.train()
    
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    print("✅ Training completed!")

if __name__ == "__main__":
    train()
