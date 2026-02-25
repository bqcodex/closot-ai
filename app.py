from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import os

PRODUCT_CONTEXT = """
You are an AI assistant embedded inside closot, a productivity and collaboration application.
"""

app = FastAPI()

MODEL_NAME = "Qwen/Qwen3-8B"

# ===== API KEY CONFIG =====
API_KEY = os.getenv("LLM_API_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-stream")
async def generate_stream(
    request: PromptRequest,
    _: str = Depends(verify_api_key)   # 👈 PROTECTED
):

    messages = [
        {"role": "system", "content": PRODUCT_CONTEXT},
        {"role": "user", "content": request.prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=1024
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_generator():
        for new_text in streamer:
            yield new_text

    return StreamingResponse(token_generator(), media_type="text/plain")