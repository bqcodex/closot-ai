from fastapi import Request , FastAPI, Depends, HTTPException, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
from typing import Annotated



app = FastAPI()


API_KEY = os.getenv("LLM_API_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_api_key(api_key: Annotated[str, api_key_header]):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

MODEL_PATH = "qwen25-3b-4bit"
model, tokenizer = load(MODEL_PATH)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-stream")
async def generate_stream(request: Request, api_key: Annotated[str, api_key_header] = Depends(verify_api_key)):
    raw_body = await request.body()
    prompt = raw_body.decode('utf-8')
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.06)

    full_response = ""
    for token in generate(model, tokenizer, prompt, sampler=sampler, max_tokens=32768, verbose=False):
        full_response += token

    return Response(content=full_response, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,timeout_keep_alive=300,limit_max_request_size=0)