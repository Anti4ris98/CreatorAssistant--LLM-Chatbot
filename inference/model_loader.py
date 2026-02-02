import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ChatModel:
    def __init__(self, base_model_name, finetuned_path):
        print("🔄 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.model = PeftModel.from_pretrained(base_model, finetuned_path)
        self.model.eval()
        
        print("✅ Model ready!")
    
    def generate_response(self, prompt, max_length=256, temperature=0.7):
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
