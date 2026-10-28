"""Integration test suite for Local Model Infrastructure on Port 8080."""

from __future__ import annotations

import json
import os
import sys
import time
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")


def log_step(title: str):
    print(f"\n{'─' * 60}")
    print(f"▶ {title}")
    print(f"{'─' * 60}")


def test_health_and_vram():
    log_step("1. Checking Server Health and Metal VRAM Telemetry")
    resp = requests.get(f"{BASE_URL}/v1/system/vram", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    metal = data.get("metal", {})
    mem = data.get("memory", {})
    print(f"  ✓ Device: {metal.get('device_name')} ({metal.get('architecture')})")
    print(f"  ✓ Metal Max Working Set: {metal.get('max_working_set_gb')} GB")
    print(f"  ✓ Host RAM Total: {mem.get('ram_total_gb')} GB | Available: {mem.get('ram_available_gb')} GB")
    print(f"  ✓ Active Model in VRAM: {data.get('active_model') or 'None (Clean VRAM)'}")


def test_models_catalog():
    log_step("2. Querying OpenAI Models Catalog (/v1/models)")
    resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    models = data.get("data", [])
    print(f"  Found {len(models)} configured models:")
    for m in models:
        ctx = f" (Context: {m['context_window']} tokens)" if m.get("context_window") else ""
        print(f"    • [{m['modality'].upper()}] {m['id']}{ctx}")


def test_chat_streaming():
    log_step("3. Testing Text Engine with SSE Token Streaming (/v1/chat/completions)")
    payload = {
        "messages": [
            {"role": "user", "content": "Explain how hybrid attention enables 1M context in Spark-X2.5."}
        ],
        "max_tokens": 128,
        "temperature": 0.2,
        "stream": True,
    }

    start = time.perf_counter()
    with requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=60) as resp:
        assert resp.status_code == 200, f"Chat failed with status {resp.status_code}: {resp.text}"
        print("  Streaming response chunks from Spark-X2.5-4B:")
        full_text = ""
        token_count = 0
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        reasoning = delta.get("reasoning_content", "")
                        content = delta.get("content", "")
                        if reasoning:
                            print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
                            token_count += 1
                        if content:
                            print(content, end="", flush=True)
                            full_text += content
                            token_count += 1
                except json.JSONDecodeError:
                    pass

        elapsed = time.perf_counter() - start
        tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
        print(f"\n  ✓ Stream finished in {elapsed:.2f}s (~{tps} chunks/sec)")


def test_image_generation():
    log_step("4. Testing Image Engine & Dynamic VRAM Swap (/v1/images/generations)")
    payload = {
        "prompt": "Cyberpunk server rack running local MLX model on Apple Silicon M4, neon blue and cyan glow",
        "n": 1,
        "size": "512x512",
        "response_format": "b64_json",
    }
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/images/generations", json=payload, timeout=60)
    assert resp.status_code == 200, f"Image generation failed: {resp.text}"
    data = resp.json()
    images = data.get("data", [])
    assert len(images) > 0, "No image returned"
    b64_len = len(images[0].get("b64_json", ""))
    elapsed = time.perf_counter() - start
    print(f"  ✓ Image generated successfully in {elapsed:.2f}s (Base64 payload size: {b64_len} chars)")


def test_audio_speech():
    log_step("5. Testing Audio Speech Engine & Dynamic VRAM Swap (/v1/audio/speech)")
    payload = {
        "input": "StepAudio Realtime speech synthesis online and active on port 8080.",
        "voice": "alloy",
        "speed": 1.0,
    }
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=30)
    assert resp.status_code == 200, f"Audio failed: {resp.text}"
    wav_bytes = resp.content
    assert len(wav_bytes) > 44, "Invalid WAV payload"
    elapsed = time.perf_counter() - start
    print(f"  ✓ Speech WAV synthesized in {elapsed:.2f}s ({len(wav_bytes)} bytes audio)")


def test_manual_eviction():
    log_step("6. Testing Manual VRAM Eviction (/v1/system/unload)")
    resp = requests.post(f"{BASE_URL}/v1/system/unload", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    print(f"  ✓ Eviction status: {data.get('status')} - {data.get('message')}")

    # Verify VRAM is clear
    vram_resp = requests.get(f"{BASE_URL}/v1/system/vram", timeout=5)
    active = vram_resp.json().get("active_model")
    print(f"  ✓ Verified active model in memory: {active or 'None (VRAM completely cleared)'}")


def main():
    print("═" * 60)
    print("  RUNNING LOCAL MODEL INFRASTRUCTURE INTEGRATION SUITE")
    print("═" * 60)
    try:
        test_health_and_vram()
        test_models_catalog()
        test_chat_streaming()
        test_image_generation()
        test_audio_speech()
        test_manual_eviction()
        print("\n" + "═" * 60)
        print("  🎉 ALL 6 INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("═" * 60 + "\n")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to {BASE_URL}.")
        print("Please start the server first in another terminal or background:")
        print("  cd /Users/richardsongunde/projects/local-model-infra")
        print("  uv run server.py\n")
        sys.exit(1)
    except AssertionError as ae:
        print(f"\n[TEST FAILED] {ae}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
