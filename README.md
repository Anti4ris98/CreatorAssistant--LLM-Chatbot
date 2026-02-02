# 🤖 CreatorAssistant: Fine-Tuned LLM Chatbot for Creator Economy

End-to-end PyTorch implementation of a conversational AI chatbot fine-tuned for creator-fan interactions.

## 🎯 Project Overview

This project demonstrates:
- Fine-tuning LLMs with **LoRA/QLoRA** on custom datasets
- Building production-ready **inference API** with FastAPI
- Creating interactive **UI** with Streamlit
- Implementing **evaluation pipeline** for model quality

## 🛠️ Tech Stack

- **Model**: Llama-2-7B-chat
- **Framework**: PyTorch + Transformers + PEFT
- **Training**: QLoRA (4-bit quantization)
- **API**: FastAPI
- **UI**: Streamlit
- **Dataset**: Bitext Customer Support (adapted for creator economy)

## 📊 Results

- **Training samples**: ~24,000
- **Training time**: ~3 hours on T4 GPU
- **Model size**: ~3GB (with LoRA adapters)
- **Inference speed**: ~2-3 sec per response

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python data/prepare_data.py
```

### 3. Train model
```bash
python training/train.py
```

**Note**: Training requires a GPU. You can use:
- Google Colab (free T4 GPU)
- Kaggle Notebooks (free GPU)
- Local GPU (NVIDIA with CUDA support)

Expected training time: 2-4 hours on T4 GPU

### 4. Run API server
```bash
python inference/app.py
```

The API will be available at `http://localhost:8000`

### 5. Launch UI
```bash
streamlit run ui/streamlit_app.py
```

The Streamlit interface will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
CreatorAssistant/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── training_config.yaml        # Training hyperparameters
├── data/
│   ├── raw/                        # Original dataset (auto-downloaded)
│   ├── processed/                  # Processed train/eval splits
│   │   ├── train.json
│   │   └── eval.json
│   └── prepare_data.py             # Data preparation script
├── training/
│   ├── train.py                    # Main training script
│   ├── model_config.py             # Model configuration classes
│   └── utils.py                    # Training utilities
├── inference/
│   ├── app.py                      # FastAPI server
│   ├── chat_handler.py             # Chat history management
│   └── model_loader.py             # Model loading utilities
├── ui/
│   └── streamlit_app.py            # Streamlit interface
├── evaluation/
│   ├── evaluate.py                 # Evaluation script
│   ├── test_prompts.json           # Test questions
│   └── results/                    # Evaluation results
│       └── comparison.csv
├── models/
│   ├── base/                       # Base model (auto-downloaded)
│   └── finetuned/                  # Fine-tuned model with LoRA adapters
└── logs/
    └── training.log                # Training logs
```

## 🔧 Configuration

Edit `config/training_config.yaml` to customize:

- **Model selection**: Choose between Llama-2, Mistral, or other models
- **LoRA parameters**: Adjust `r`, `lora_alpha`, `target_modules`
- **Training hyperparameters**: Batch size, learning rate, epochs, etc.

## 📚 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Endpoint
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I cancel my membership?",
    "temperature": 0.7,
    "max_length": 256
  }'
```

Response:
```json
{
  "response": "To cancel your membership, please visit your account settings..."
}
```

## 🧪 Evaluation

Run evaluation on test prompts:

```bash
python evaluation/evaluate.py
```

This will:
1. Load the fine-tuned model
2. Test it on predefined prompts
3. Generate response statistics
4. Save results to `evaluation/results/comparison.csv`

## 🎓 Key Learnings

- ✅ Implemented LoRA for parameter-efficient fine-tuning
- ✅ Built end-to-end ML pipeline from data to deployment
- ✅ Optimized inference for production use
- ✅ Created evaluation framework for LLM quality

## 🔄 Next Steps

Potential improvements:
- [ ] Add RAG (Retrieval-Augmented Generation) for knowledge base
- [ ] Implement conversation memory across sessions
- [ ] Add user authentication to the API
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add response quality metrics (BLEU, ROUGE, etc.)
- [ ] Implement A/B testing framework
- [ ] Add multi-language support

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

Made with ❤️ for the Creator Economy
