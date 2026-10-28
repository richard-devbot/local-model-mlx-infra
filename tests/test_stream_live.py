#!/usr/bin/env python3
"""Interactive demo script showing live thinking token streaming via SSE."""

import json
import sys
import time
import requests

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

def stream_prompt(prompt: str, max_tokens: int = 400) -> None:
    print("\n" + "=" * 65)
    print(f"User Prompt: {prompt}")
    print("=" * 65 + "\n")

    payload = {
        "model": "hermitdave/Spark-X2.5-4B-MLX-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    in_thinking = True
    thinking_chars = 0
    content_chars = 0

    print("🧠 Thinking (Live): ", end="", flush=True)

    with requests.post(API_URL, json=payload, stream=True, timeout=60) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                reasoning_chunk = delta.get("reasoning_content")
                content_chunk = delta.get("content")

                if reasoning_chunk:
                    # Dim / gray thinking text
                    print(f"\033[90m{reasoning_chunk}\033[0m", end="", flush=True)
                    thinking_chars += len(reasoning_chunk)

                if content_chunk:
                    if in_thinking:
                        in_thinking = False
                        print("\n\n💬 Response: ", end="", flush=True)
                    print(content_chunk, end="", flush=True)
                    content_chars += len(content_chunk)

            except json.JSONDecodeError:
                pass

    dur = time.time() - t0
    print(f"\n\n⏱ Finished in {dur:.2f}s | Thinking: {thinking_chars} chars | Response: {content_chars} chars\n")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "Solve this: A bat and a ball cost $1.10 in total. "
        "The bat costs $1.00 more than the ball. How much does the ball cost?"
    )
    stream_prompt(prompt)
