"""System telemetry, model enumeration, and VRAM management routes."""

from __future__ import annotations

import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.arbiter import arbiter
from app.config import settings
from app.system import get_full_telemetry

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_model": arbiter.active_model_id,
        "active_modality": arbiter.active_modality,
        "timestamp": int(time.time()),
    }


@router.get("/v1/models")
async def list_models():
    """OpenAI-compatible models catalog endpoint."""
    active_id = arbiter.active_model_id

    models_data = [
        {
            "id": settings.text_model_bf16 if settings.prefer_bf16_text else settings.text_model,
            "object": "model",
            "created": 1720000000,
            "owned_by": "local-model-infra",
            "modality": "text",
            "context_window": 1048576,  # 1M context
            "active_in_vram": active_id == (
                settings.text_model_bf16 if settings.prefer_bf16_text else settings.text_model
            ),
            "description": "Spark-X2.5-4B with 1M context window and hybrid attention",
        },
        {
            "id": settings.image_model,
            "object": "model",
            "created": 1720000000,
            "owned_by": "local-model-infra",
            "modality": "image",
            "context_window": None,
            "active_in_vram": active_id == settings.image_model,
            "description": "Qwen-Image-2.1 unified text-to-image and image editing",
        },
        {
            "id": settings.audio_model,
            "object": "model",
            "created": 1720000000,
            "owned_by": "local-model-infra",
            "modality": "audio",
            "context_window": None,
            "active_in_vram": active_id == settings.audio_model,
            "description": "StepAudio 3 Realtime & Step-Audio-TTS-3B full-duplex speech",
        },
    ]

    return JSONResponse(
        content={
            "object": "list",
            "data": models_data,
        }
    )


@router.get("/v1/system/vram")
async def system_vram_telemetry():
    """Apple Metal VRAM and host memory monitoring."""
    telemetry = get_full_telemetry(active_model=arbiter.active_model_id)
    return JSONResponse(content=telemetry)


@router.post("/v1/system/unload")
async def unload_active_model():
    """Manually unpin the active model from VRAM and free Apple Metal memory."""
    active = arbiter.active_model_id
    if not active:
        return {"status": "noop", "message": "No model is currently loaded in memory."}

    await arbiter.unload_active()
    return {
        "status": "success",
        "unloaded_model": active,
        "message": "Model evicted and Metal memory cache cleared.",
    }
