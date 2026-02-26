from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os

app = FastAPI()

API_KEY = os.getenv("LLM_API_KEY", "super-secret-key")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

MODEL_PATH = "/Users/mac/ai/models/qwen3-8b-4bit"
model, tokenizer = load(MODEL_PATH)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-stream")
async def generate_stream(request: PromptRequest, _: str = Depends(verify_api_key)):
    messages = [{"role": "user", "content": request.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  
    )
    
    sampler = make_sampler(temp=0.1, top_p=0.95)
    
    def stream_tokens():  
        for token in generate(
            model, tokenizer, prompt=prompt,
            max_tokens=512,
            sampler=sampler,
            verbose=False
        ):
            yield token  
    
    return StreamingResponse(stream_tokens(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
