"""Tests for tool calling, multi-format parsing, and SSE streaming."""

from __future__ import annotations

import json
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from app.engines.text_spark import SparkTextEngine
from server import app


@pytest.fixture
def engine():
    return SparkTextEngine()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# =========================================================================
# 1. MULTI-FORMAT TOOL EXTRACTION TESTS
# =========================================================================

def test_extract_spark_xml_tool_call(engine: SparkTextEngine):
    raw = (
        "I will calculate the mortgage for you.\n"
        "<tool_call>calculate_mortgage"
        "<arg_key>principal</arg_key><arg_value>500000</arg_value>"
        "<arg_key>annual_rate</arg_key><arg_value>6.5</arg_value>"
        "<arg_key>years</arg_key><arg_value>30</arg_value>"
        "</tool_call>"
    )
    clean, calls = engine.extract_tool_calls(raw)
    assert clean == "I will calculate the mortgage for you."
    assert len(calls) == 1
    call = calls[0]
    assert call["function"]["name"] == "calculate_mortgage"
    args = json.loads(call["function"]["arguments"])
    assert args["principal"] == 500000
    assert args["annual_rate"] == 6.5
    assert args["years"] == 30


def test_extract_hermes_json_tool_call(engine: SparkTextEngine):
    raw = (
        "Let me run this command.\n"
        "<tool_call>\n"
        '{"name": "bash", "arguments": {"command": "ls -la /tmp"}}\n'
        "</tool_call>"
    )
    clean, calls = engine.extract_tool_calls(raw)
    assert clean == "Let me run this command."
    assert len(calls) == 1
    call = calls[0]
    assert call["function"]["name"] == "bash"
    args = json.loads(call["function"]["arguments"])
    assert args["command"] == "ls -la /tmp"


def test_extract_hermes_json_with_string_args(engine: SparkTextEngine):
    raw = (
        "<tool_call>\n"
        '{"name": "read_file", "arguments": "{\\"path\\": \\"/etc/hosts\\"}"}\n'
        "</tool_call>"
    )
    clean, calls = engine.extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["path"] == "/etc/hosts"


def test_extract_hermes_parameters_key(engine: SparkTextEngine):
    raw = (
        "<tool_call>\n"
        '{"name": "get_weather", "parameters": {"city": "Tokyo", "unit": "celsius"}}\n'
        "</tool_call>"
    )
    _, calls = engine.extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["city"] == "Tokyo"
    assert args["unit"] == "celsius"


def test_extract_xml_attribute_tool_call(engine: SparkTextEngine):
    raw = (
        '<tool_call name="fetch_url">\n'
        '{"url": "https://example.com"}\n'
        "</tool_call>"
    )
    _, calls = engine.extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "fetch_url"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["url"] == "https://example.com"


def test_extract_multiple_parallel_tool_calls(engine: SparkTextEngine):
    raw = (
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>file1.txt</arg_value></tool_call>"
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>file2.txt</arg_value></tool_call>"
    )
    clean, calls = engine.extract_tool_calls(raw)
    assert clean == ""
    assert len(calls) == 2
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "file1.txt"
    assert calls[1]["id"] == "call_2"
    assert calls[1]["function"]["name"] == "read_file"
    assert json.loads(calls[1]["function"]["arguments"])["path"] == "file2.txt"


def test_extract_unclosed_tool_call(engine: SparkTextEngine):
    raw = "<tool_call>read_file<arg_key>path</arg_key><arg_value>/var/log/app.log</arg_value>"
    clean, calls = engine.extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "/var/log/app.log"


def test_extract_markdown_json_fallback(engine: SparkTextEngine):
    tools = [{"type": "function", "function": {"name": "execute_code"}}]
    raw = (
        "I will execute this:\n"
        "```json\n"
        '{"name": "execute_code", "arguments": {"code": "print(123)"}}\n'
        "```"
    )
    clean, calls = engine.extract_tool_calls(raw, available_tools=tools)
    assert clean == "I will execute this:"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "execute_code"


# =========================================================================
# 2. MESSAGE NORMALIZATION & PARSING TESTS
# =========================================================================

def test_normalize_messages_for_jinja(engine: SparkTextEngine):
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "What is the time?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',  # String from OpenAI client!
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": {"time": "12:00"}},  # Dict content
    ]
    norm = engine._normalize_messages_for_template(messages)
    # Check that tool call arguments were converted to a dict for Jinja template
    tc_args = norm[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(tc_args, dict)
    assert tc_args["timezone"] == "UTC"
    # Check that tool content was converted to a string
    assert isinstance(norm[2]["content"], str)


def test_parse_spark_output_with_thinking(engine: SparkTextEngine):
    raw = (
        "<think>Let me verify the parameters first.</think>"
        "I am calling the weather tool.\n"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
    )
    reasoning, clean_content, tool_calls = engine._parse_spark_output(raw)
    assert reasoning == "Let me verify the parameters first."
    assert clean_content == "I am calling the weather tool."
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


# =========================================================================
# 3. FASTAPI ENDPOINT INTEGRATION TESTS
# =========================================================================

def test_chat_non_stream_with_tools(client: TestClient):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get stock price for ticker",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        }
    ]
    payload = {
        "messages": [{"role": "user", "content": "What is the stock price of AAPL?"}],
        "tools": tools,
        "stream": False,
        "max_tokens": 128,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    # Check finish_reason
    assert choice["finish_reason"] in ("tool_calls", "stop")
    if choice["finish_reason"] == "tool_calls":
        assert "tool_calls" in choice["message"]
        assert len(choice["message"]["tool_calls"]) > 0
        call = choice["message"]["tool_calls"][0]
        assert call["type"] == "function"
        assert isinstance(call["function"]["arguments"], str)


def test_chat_streaming_with_tools(client: TestClient):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_db",
                "description": "Search database",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    payload = {
        "messages": [{"role": "user", "content": "Search for user id 123"}],
        "tools": tools,
        "stream": True,
        "max_tokens": 128,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    chunks = []
    finish_reason = None
    for line in resp.text.split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            choice = chunk["choices"][0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    assert len(chunks) > 0
    assert finish_reason in ("tool_calls", "stop")


def test_chat_multi_turn_history_with_tool_call(client: TestClient):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Calculating...",
            "tool_calls": [
                {
                    "id": "call_calc_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_calc_1",
            "name": "calculator",
            "content": "4",
        },
    ]
    payload = {
        "messages": messages,
        "stream": False,
        "max_tokens": 128,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
