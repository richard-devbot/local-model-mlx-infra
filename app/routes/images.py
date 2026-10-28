"""Image generation route supporting OpenAI /v1/images/generations format."""

from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.arbiter import arbiter
from app.config import settings

router = APIRouter(prefix="/v1", tags=["images"])


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=4)
    size: str = Field(default="1024x1024")
    response_format: str = Field(default="b64_json")
    steps: int = Field(default=25, ge=1, le=100)


@router.post("/images/generations")
async def create_image_generation(request: ImageGenerationRequest):
    """OpenAI-compatible image generation endpoint powered by Qwen-Image-2.1."""
    target_model = request.model or settings.image_model

    try:
        engine = await arbiter.acquire_engine("image", target_model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to acquire image engine: {exc}")

    try:
        # Run generation in worker thread to keep server responsive
        response_data = await asyncio.to_thread(
            engine.generate,  # type: ignore
            prompt=request.prompt,
            n=request.n,
            size=request.size,
            response_format=request.response_format,
            steps=request.steps,
        )
        return JSONResponse(content=response_data)
    except Exception as gen_err:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {gen_err}")
