from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import ChatModel
import uvicorn

app = FastAPI(title="CreatorAssistant API")

# Загрузка модели при старте
chat_model = ChatModel(
    base_model_name="meta-llama/Llama-2-7b-chat-hf",
    finetuned_path="./models/finetuned"
)

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_length: int = 256

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chat_model.generate_response(
            prompt=request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
