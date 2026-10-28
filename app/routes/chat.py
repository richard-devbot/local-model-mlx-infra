"""Chat completions route supporting OpenAI protocol with SSE streaming."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.arbiter import arbiter
from app.config import settings

router = APIRouter(prefix="/v1", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint with SSE streaming."""
    target_model = request.model or (
        settings.text_model_bf16 if settings.prefer_bf16_text else settings.text_model
    )

    try:
        # Acquire text engine (swapping out previous model if needed)
        engine = await arbiter.acquire_engine("text", target_model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    messages_payload: List[Dict[str, Any]] = []
    for m in request.messages:
        content_val = m.content
        if isinstance(content_val, list):
            parts = []
            for b in content_val:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        parts.append(b.get("text", ""))
                    elif "text" in b:
                        parts.append(str(b["text"]))
                elif isinstance(b, str):
                    parts.append(b)
            content_val = "".join(parts)
        elif content_val is None:
            content_val = ""

        item: Dict[str, Any] = {"role": m.role, "content": content_val}
        if m.name:
            item["name"] = m.name
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        if m.reasoning_content:
            item["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            normalized_calls = []
            for idx, tc in enumerate(m.tool_calls):
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                fn["arguments"] = args
                normalized_calls.append({
                    "id": tc.get("id") or f"call_{idx+1}",
                    "type": "function",
                    "function": fn,
                })
            item["tool_calls"] = normalized_calls
        messages_payload.append(item)
    if request.stream:
        async def event_generator():
            async with arbiter.track_activity():
                try:
                    async for chunk in engine.generate_stream(  # type: ignore
                        messages=messages_payload,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop,
                    ):
                        arbiter.touch()
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as stream_err:
                    error_chunk = {
                        "error": {
                            "message": str(stream_err),
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    async with arbiter.track_activity():
        result = await engine.generate_non_stream(  # type: ignore
            messages=messages_payload,
            tools=request.tools,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
    return JSONResponse(content=result)
