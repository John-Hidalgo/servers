import os
import base64
import httpx
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLM_URL = os.getenv("VLM_URL", "http://qwen2-5-vl:8080")

def load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "analyze.txt")
    with open(prompt_path, "r") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_prompt()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AnalyzeURLRequest(BaseModel):
    image_url: str
    custom_prompt: Optional[str] = None

class ImageAnalysis(BaseModel):
    object: str
    colours: list[str]
    materials: list[str]
    style: str

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VLM Wrapper", description="Qwen2.5-VL image analysis API")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def call_vlm(image_content: str, content_type: str, system_prompt: str) -> dict:
    """
    Calls the llama.cpp /v1/chat/completions endpoint with an image.
    image_content is either a base64 string or a URL.
    cache_prompt=true so the system prompt tokens are KV-cached across requests.
    """
    if content_type == "base64":
        image_url_value = f"data:image/jpeg;base64,{image_content}"
    else:
        image_url_value = image_content

    payload = {
        "model": "qwen2.5-vl",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_value}
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image and return the JSON."
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,       # low temp for consistent structured output
        "cache_prompt": True      # KV cache the system prompt across requests
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(f"{VLM_URL}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    raw = data["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if model wraps output despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail={"error": "Model returned invalid JSON", "raw": raw}
        )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/analyze/upload", response_model=ImageAnalysis)
async def analyze_upload(file: UploadFile = File(...)):
    """Accept an image file upload and return structured analysis."""
    contents = await file.read()
    b64 = base64.b64encode(contents).decode("utf-8")
    result = await call_vlm(b64, "base64", SYSTEM_PROMPT)
    return result


@app.post("/analyze/url", response_model=ImageAnalysis)
async def analyze_url(request: AnalyzeURLRequest):
    """Accept a public image URL and return structured analysis."""
    prompt = request.custom_prompt or SYSTEM_PROMPT
    result = await call_vlm(request.image_url, "url", prompt)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "vlm_url": VLM_URL}