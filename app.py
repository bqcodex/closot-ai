import asyncio
import os
import copy
from contextlib import asynccontextmanager
from fastapi import Request , FastAPI, Depends, HTTPException, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
from typing import Annotated
from mlx_lm import load, stream_generate
with open("agent-blue-print.txt", "r") as f:
    AGENT_BLUEPRINT = f.read()

def get_dynamic_context():

    return """
## CURRENT WORKSPACE STATE
- CURRENT_PAGE_ID: "__currentPageId__"
- WORKSPACE_MEMBERS: ["Nikita", "Athav", "Tarun"]
- ACTIVE_WORKAREAS: ["Engineering", "Design"]
"""

# --- 2. LIFESPAN & STATE MANAGEMENT ---

model = None
tokenizer = None
SYSTEM_CACHE = []
cache_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, SYSTEM_CACHE

    MODEL_PATH = "qwen25-3b-4bit"
    model, tokenizer = load(
        MODEL_PATH,
        model_config={"kv_bits": 4, "kv_group_size": 64}
    )

    # --- THE ROBUST CACHE INITIALIZATION ---
    # We attempt to find the KVCache class in the three most common locations
    KVCache = None
    import_paths = [
        "mlx_lm.models.cache.KVCache",
        "mlx_lm.models.base.KVCache",
        "mlx_lm.utils.KVCache"
    ]

    import importlib
    for path in import_paths:
        try:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            KVCache = getattr(module, class_name)
            break
        except (ImportError, AttributeError):
            continue

    if KVCache is None:
        raise RuntimeError("Could not locate KVCache class in mlx_lm")

    # Initialize the list with real KVCache objects for each layer
    # This prevents the 'IndexError: list index out of range'
    num_layers = len(model.layers)
    SYSTEM_CACHE = [KVCache() for _ in range(num_layers)]

    full_system_prompt = AGENT_BLUEPRINT + get_dynamic_context()
    messages = [{"role": "system", "content": full_system_prompt}]
    prefill_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # PREFILL: Process the blueprint into the SYSTEM_CACHE
    for _ in stream_generate(model, tokenizer, prompt=prefill_text, prompt_cache=SYSTEM_CACHE, max_tokens=1):
        break

    print(f"✅ Background Agent Initialized. Cached {num_layers} layers.")
    yield
    SYSTEM_CACHE = None
# --- 3. API SETUP ---

app = FastAPI(lifespan=lifespan)

API_KEY = os.getenv("LLM_API_KEY")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def verify_api_key(api_key: Annotated[str, api_key_header]):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-stream")
async def generate_stream(request: Request, api_key: Annotated[str, api_key_header] = Depends(verify_api_key)):
    async with cache_lock:
        raw_body = await request.body()
        prompt = raw_body.decode('utf-8')
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if "<|im_start|>user" in prompt:
             prompt = "<|im_start|>user" + prompt.split("<|im_start|>user")[-1]
    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.06)
    request_cache = list(SYSTEM_CACHE)
    full_response = ""
    for token in generate(model, tokenizer, prompt, prompt_cache=request_cache, sampler=sampler, max_tokens=32768, verbose=False):
        full_response += token

    return Response(content=full_response, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000,timeout_keep_alive=300,limit_max_request_size=0)