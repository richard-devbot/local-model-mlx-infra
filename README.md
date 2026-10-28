# Local Model Infrastructure (Apple Silicon M4)

High-performance, production-grade local AI infrastructure engineered for Apple Silicon (M-series) Macs with unified memory management.

Exposes a unified server on **port 8080** serving three modalities through OpenAI-compatible endpoints:
1. **Text (1M Context Window):** `XHToken/Spark-X2.5-4B` / `hermitdave/Spark-X2.5-4B-MLX-4bit` (MLX native streaming SSE)
2. **Image Generation:** `Qwen/Qwen-Image-2.1` / `mlx-community/Qwen-Image-2.1-MLX-4bit`
3. **Realtime Audio:** `StepAudio 3 Realtime` / `stepfun-ai/Step-Audio-TTS-3B`

---

## Architecture: Unified Memory VRAM Arbiter

On a 16 GB Apple Silicon Mac (such as an M4 Mac mini), unified memory is shared between macOS, CPU, and the Metal GPU. macOS recommends a working set ceiling of **~11.84 GB** (`max_recommended_working_set_size`).

Loading multiple multi-gigabyte models simultaneously causes out-of-memory crashes or heavy disk swap. `local-model-infra` solves this with an **intelligent on-demand VRAM Arbiter**:

```
                       Client Requests (Port 8080)
      POST /v1/chat/completions | POST /v1/images/generations | POST /v1/audio/speech
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │    FastAPI Gateway (:8080)    │
                   └──────────────┬────────────────┘
                                  │
                   ┌─���────────────▼────────────────┐
                   │     VRAM Arbiter & Lock       │
                   │  - Single active model lock   │
                   │  - On-demand dynamic loading  │
                   │  - mx.metal.clear_cache()     │
                   │  - torch.mps.empty_cache()    │
                   │  - Idle eviction (TTL: 300s)  │
                   └──────────────┬────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ Spark-X2.5-4B     │   │ Qwen-Image-2.1    │   │ StepAudio 3       │
│ Precision: BF16   │   │ Precision: 4-bit  │   │ Precision: 4-bit  ���
│ 1M Token Context  │   │ Sequential/MLX    │   │ Realtime Stream   │
│ SSE Token Stream  │   │ Base64/PNG Out    │   │ PCM/WAV Streaming │
│ ~2.5 - 8.2 GB     │   │ ~5.5 GB           │   │ ~3.5 GB           │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

### Safety Guarantees
* **Zero GPU degradation:** Weights stored on disk (SSD) never touch the GPU until requested.
* **Metal Memory Flush:** Evicting a model triggers `del model`, `gc.collect()`, and `mx.metal.clear_cache()`, immediately returning memory to macOS.
* **Auto-eviction:** Idle models unpin from VRAM after 5 minutes (configurable) of inactivity.

---

## Quickstart

Run with `uv`:

```bash
cd /Users/richardsongunde/projects/local-model-infra

# Run the server
uv run server.py

# In another terminal, run the test suite
uv run test-infra.py
```

---

## API Endpoints

### 1. Chat Completions (SSE Streaming)
* **Endpoint:** `POST http://127.0.0.1:8080/v1/chat/completions`
* **Format:** OpenAI Compatible
* **Model:** `hermitdave/Spark-X2.5-4B-MLX-4bit` (or `XHToken/Spark-X2.5-4B`)

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hermitdave/Spark-X2.5-4B-MLX-4bit",
    "messages": [{"role": "user", "content": "Explain hybrid attention in 1 sentence."}],
    "stream": true
  }'
```

#### Agent & Tool Calling Integration (Hermes Agent / Aider / OpenAI SDK)
The endpoint provides production OpenAI function calling with **true SSE streaming** (`delta.tool_calls` chunks and `finish_reason: "tool_calls"`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

response = client.chat.completions.create(
    model="hermitdave/Spark-X2.5-4B-MLX-4bit",
    messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    stream=True,  # Full streaming delta.tool_calls support
)

for chunk in response:
    if chunk.choices[0].delta.tool_calls:
        print("Tool call delta:", chunk.choices[0].delta.tool_calls)
```

### 2. Image Generation
* **Endpoint:** `POST http://127.0.0.1:8080/v1/images/generations`
* **Format:** OpenAI Compatible
* **Model:** `mlx-community/Qwen-Image-2.1-MLX-4bit`

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Futuristic control room in neon blue style",
    "size": "512x512",
    "response_format": "b64_json"
  }'
```

### 3. Speech Synthesis & Streaming
* **Endpoint:** `POST http://127.0.0.1:8080/v1/audio/speech`
* **Realtime Streaming:** `POST http://127.0.0.1:8080/v1/audio/duplex_stream`
* **Model:** `stepfun-ai/Step-Audio-TTS-3B`

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from StepAudio on Apple Silicon!"}' \
  --output test.wav
```

### 4. Telemetry and VRAM Status
* **Status:** `GET http://127.0.0.1:8080/v1/system/vram`
* **Models:** `GET http://127.0.0.1:8080/v1/models`
* **Force Evict:** `POST http://127.0.0.1:8080/v1/system/unload`

---

## Configuration Flags

| CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | `MODEL_PORT` | `8080` | Bind port |
| `--host` | `MODEL_HOST` | `127.0.0.1` | Bind host |
| `--prefer-bf16` | `PREFER_BF16_TEXT` | `false` | Load Spark-X2.5 in unquantized BF16 |
| `--idle-ttl` | `MODEL_IDLE_TTL_SECONDS` | `300` | Evict idle model after N seconds (0 to disable) |
| `--text-model` | `LOCAL_TEXT_MODEL` | `hermitdave/Spark-X2.5-4B-MLX-4bit` | Text model repository |
| `--image-model` | `LOCAL_IMAGE_MODEL` | `mlx-community/Qwen-Image-2.1-MLX-4bit` | Image model repository |
| `--audio-model` | `LOCAL_AUDIO_MODEL` | `stepfun-ai/Step-Audio-TTS-3B` | Audio model repository |

---

## Agentic Benchmarks & Live Streaming

The infrastructure includes comprehensive evaluation suites covering all 5 agent pillars:

```bash
# Run all benchmark suites
uv run test_benchmarks.py

# Or run targeted individual suites:
uv run test_benchmarks.py --suite reasoning   # GSM8K, Deductive Logic, DAG Planning
uv run test_benchmarks.py --suite tools       # BFCL Selection, Typed Args, No-Tool Restraint
uv run test_benchmarks.py --suite context     # Single-Needle (2K, 8K) & Multi-Needle Cross-Doc
uv run test_benchmarks.py --suite streaming   # Live SSE delta.reasoning_content Protocol
uv run test_benchmarks.py --suite search      # End-to-End Live Web Search Agent Loop

# Test interactive live thinking stream in terminal
uv run test_stream_live.py "Solve this: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
```
