import os
import httpx
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

def load_default_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "queries_prompt.txt")
    with open(prompt_path, 'r') as f:
        return f.read().strip()

DEFAULT_SYSTEM_PROMPT = load_default_prompt()
LLAMA_URL = "http://llama-cpp:8080"

class SpaceRequest(BaseModel):
    space_description: str

class CustomPromptRequest(BaseModel):
    system_prompt: str
    space_description: str

app = FastAPI()

async def call_llama(system_prompt: str, user_message: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

@app.post("/space/objects")
async def get_space_objects(request: SpaceRequest):
    response_text = await call_llama(DEFAULT_SYSTEM_PROMPT, request.space_description)
    
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        try:
            items = json.loads(json_match.group())
            return {"items": items, "space": request.space_description}
        except json.JSONDecodeError:
            pass
    
    return {"raw_response": response_text, "space": request.space_description}

@app.post("/space/objects/custom")
async def get_space_objects_custom(request: CustomPromptRequest):
    response_text = await call_llama(request.system_prompt, request.space_description)
    return {"response": response_text, "space": request.space_description}

@app.get("/health")
async def health_check():
    return {"status": "ok"}