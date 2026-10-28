"""Test OpenAI Python SDK client interaction (streaming and non-streaming tool calling)."""

from __future__ import annotations

import json
import httpx
import pytest
from openai import AsyncOpenAI

from server import app


@pytest.mark.anyio
async def test_openai_sdk_non_streaming_tool_call():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/v1") as http_client:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="local-token",
            http_client=http_client,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Lookup current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        # 1. First turn: Model makes tool call
        response = await client.chat.completions.create(
            model="hermitdave/Spark-X2.5-4B-MLX-4bit",
            messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
            tools=tools,
            temperature=0.0,
            max_tokens=150,
        )

        assert response.choices[0].finish_reason in ("tool_calls", "stop")
        msg = response.choices[0].message
        assert msg.role == "assistant"

        if response.choices[0].finish_reason == "tool_calls":
            assert msg.tool_calls is not None
            tc = msg.tool_calls[0]
            assert tc.type == "function"
            assert tc.function.name == "lookup_weather"
            args = json.loads(tc.function.arguments)
            assert "city" in args

            # 2. Second turn: Feed back tool result
            conversation = [
                {"role": "user", "content": "What is the weather in Berlin?"},
                msg.model_dump(exclude_none=True),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"temperature": 18, "condition": "Sunny"}),
                },
            ]

            response2 = await client.chat.completions.create(
                model="hermitdave/Spark-X2.5-4B-MLX-4bit",
                messages=conversation,
                tools=tools,
                temperature=0.0,
                max_tokens=150,
            )
            assert response2.choices[0].finish_reason == "stop"
            assert response2.choices[0].message.content is not None
            assert len(response2.choices[0].message.content) > 0


@pytest.mark.anyio
async def test_openai_sdk_streaming_tool_call():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/v1") as http_client:
        client = AsyncOpenAI(
            base_url="http://testserver/v1",
            api_key="local-token",
            http_client=http_client,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Lookup current weather in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        stream = await client.chat.completions.create(
            model="hermitdave/Spark-X2.5-4B-MLX-4bit",
            messages=[{"role": "user", "content": "Check weather in Berlin"}],
            tools=tools,
            temperature=0.0,
            max_tokens=150,
            stream=True,
        )

        received_chunks = []
        tool_call_deltas = []
        finish_reason = None

        async for chunk in stream:
            received_chunks.append(chunk)
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if choice.delta.tool_calls:
                    tool_call_deltas.extend(choice.delta.tool_calls)

        assert len(received_chunks) > 0
        assert finish_reason in ("tool_calls", "stop")
        if finish_reason == "tool_calls":
            assert len(tool_call_deltas) > 0
            tc_delta = tool_call_deltas[0]
            assert tc_delta.function.name == "lookup_weather"
            assert "Berlin" in tc_delta.function.arguments or "berlin" in tc_delta.function.arguments.lower()
