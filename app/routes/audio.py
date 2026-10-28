"""Audio and speech synthesis routes supporting OpenAI /v1/audio format."""

from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.arbiter import arbiter
from app.config import settings

router = APIRouter(prefix="/v1", tags=["audio"])


class SpeechSynthesisRequest(BaseModel):
    input: str
    model: Optional[str] = None
    voice: str = Field(default="alloy")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


@router.post("/audio/speech")
async def create_speech(request: SpeechSynthesisRequest):
    """OpenAI-compatible text-to-speech endpoint powered by StepAudio."""
    target_model = request.model or settings.audio_model

    try:
        engine = await arbiter.acquire_engine("audio", target_model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to acquire audio engine: {exc}")

    try:
        wav_bytes = await asyncio.to_thread(
            engine.synthesize_speech_wav,  # type: ignore
            input_text=request.input,
            voice=request.voice,
            speed=request.speed,
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")


@router.post("/audio/duplex_stream")
async def stream_speech(request: SpeechSynthesisRequest):
    """Stream audio chunks for realtime speech streaming."""
    target_model = request.model or settings.audio_model

    try:
        engine = await arbiter.acquire_engine("audio", target_model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to acquire audio engine: {exc}")

    async def audio_generator():
        async for chunk in engine.stream_audio_duplex(request.input):  # type: ignore
            yield chunk

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
