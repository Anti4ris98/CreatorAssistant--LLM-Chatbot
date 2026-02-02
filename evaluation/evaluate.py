import json
import sys
sys.path.append('..')
from inference.model_loader import ChatModel
import pandas as pd
from pathlib import Path

def evaluate():
    # Create results directory
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    model = ChatModel(
        base_model_name="meta-llama/Llama-2-7b-chat-hf",
        finetuned_path="./models/finetuned"
    )
    
    # Load test prompts
    with open("evaluation/test_prompts.json", "r") as f:
        prompts = json.load(f)
    
    results = []
    
    print("🧪 Testing model...\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Testing: {prompt[:50]}...")
        response = model.generate_response(prompt)
        print(f"Response: {response[:100]}...\n")
        print("-" * 80 + "\n")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "response_length": len(response.split()),
            "response_chars": len(response)
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("evaluation/results/comparison.csv", index=False)
    
    # Statistics
    print("\n" + "="*80)
    print("EVALUATION STATISTICS")
    print("="*80)
    print(f"Total prompts tested: {len(prompts)}")
    print(f"Average response length: {df['response_length'].mean():.1f} words")
    print(f"Average response chars: {df['response_chars'].mean():.1f} characters")
    print(f"Min response length: {df['response_length'].min()} words")
    print(f"Max response length: {df['response_length'].max()} words")
    print("="*80)
    
    print(f"\n✅ Results saved to evaluation/results/comparison.csv")

if __name__ == "__main__":
    evaluate()
