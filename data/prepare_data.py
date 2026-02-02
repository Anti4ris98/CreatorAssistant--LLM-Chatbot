from datasets import load_dataset
import json
import random
from pathlib import Path

def adapt_to_creator_context(instruction, response):
    """Adapts customer support data to creator economy context"""
    
    # Replacement dictionary for creator economy
    replacements = {
        "customer": "fan",
        "product": "content",
        "order": "subscription",
        "purchase": "membership",
        "service": "creator service",
        "company": "creator",
        "support": "creator support"
    }
    
    adapted_instruction = instruction.lower()
    adapted_response = response
    
    for old, new in replacements.items():
        adapted_instruction = adapted_instruction.replace(old, new)
        adapted_response = adapted_response.replace(old, new)
    
    return adapted_instruction, adapted_response

def prepare_dataset():
    print("📥 Dataset downloading...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    
    # Creating folders
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    formatted_data = []
    
    print("🔄 Processing data...")
    for item in ds['train']:
        instruction = item['instruction']
        response = item['response']
        
        # Adapt to creator economy
        adapted_inst, adapted_resp = adapt_to_creator_context(instruction, response)
        
        # Format in Llama style
        formatted_item = {
            "instruction": adapted_inst,
            "response": adapted_resp,
            "text": f"<s>[INST] {adapted_inst} [/INST] {adapted_resp} </s>"
        }
        formatted_data.append(formatted_item)
    
    # Shuffle
    random.shuffle(formatted_data)
    
    # Split 90/10
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]
    
    # Save files
    with open("data/processed/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open("data/processed/eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Done!")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Eval samples: {len(eval_data)}")
    print(f"   Example:")
    print(f"   Instruction: {train_data[0]['instruction'][:100]}...")
    print(f"   Response: {train_data[0]['response'][:100]}...")

if __name__ == "__main__":
    prepare_dataset()
