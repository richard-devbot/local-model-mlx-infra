"""Production FastAPI Server for Local Model Infrastructure on Apple Silicon."""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.arbiter import arbiter
from app.config import settings
from app.engines.audio_step import StepAudioEngine
from app.engines.image_qwen import QwenImageEngine
from app.engines.text_spark import SparkTextEngine
from app.routes.audio import router as audio_router
from app.routes.chat import router as chat_router
from app.routes.images import router as images_router
from app.routes.system import router as system_router
from app.system import get_metal_stats, get_system_memory

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("local-infra.server")

def register_default_factories() -> None:
    """Register engine factories with the VRAM arbiter."""
    arbiter.register_factory("text", lambda model_id: SparkTextEngine(model_id))
    arbiter.register_factory("image", lambda model_id: QwenImageEngine(model_id))
    arbiter.register_factory("audio", lambda model_id: StepAudioEngine(model_id))


# Ensure factories are registered on module import
register_default_factories()
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: register engine factories and start memory monitor."""
    logger.info("Initializing Local Model Infrastructure VRAM Arbiter...")

    # Register engine factories for dynamic swapping
    register_default_factories()
    # Start auto-eviction background loop
    arbiter.start_idle_monitor()
    logger.info("Arbiter ready. Serving requests on-demand.")

    yield

    # Teardown
    logger.info("Server shutting down. Evicting active models and clearing VRAM...")
    await arbiter.stop_idle_monitor()
    await arbiter.unload_active()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Local Model Infrastructure",
        version="0.1.0",
        description="Production local model infra for Apple Silicon M4 with Unified Memory Arbiter",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(images_router)
    app.include_router(audio_router)

    return app


def print_banner(host: str, port: int) -> None:
    """Print production terminal banner with hardware details."""
    metal_info = get_metal_stats()
    sys_mem = get_system_memory()

    dev_name = metal_info.get("device_name", "Apple Silicon")
    metal_vram = metal_info.get("max_working_set_gb", "Auto")
    ram_total = sys_mem.get("ram_total_gb", 16.0)

    print("\n" + "═" * 70)
    print("  LOCAL MODEL INFRASTRUCTURE  (Apple Silicon Unified Memory Arbiter)")
    print("═" * 70)
    print(f"  Device           : {dev_name} ({metal_info.get('architecture', 'Apple GPU')})")
    print(f"  Unified Memory   : {ram_total} GB Total | Recommended Metal Limit: {metal_vram} GB")
    print(f"  Listening        : http://{host}:{port}")
    print(f"  OpenAI Base URL  : http://{host}:{port}/v1")
    print("─" * 70)
    print("  Models Configured (Loaded on-demand):")
    txt_m = settings.text_model_bf16 if settings.prefer_bf16_text else settings.text_model
    txt_prec = "Native BF16" if settings.prefer_bf16_text else "MLX 4-bit"
    print(f"    1. Text (1M Ctx) : {txt_m} [{txt_prec}]")
    print(f"    2. Image (DiT)   : {settings.image_model} [Quantized]")
    print(f"    3. Audio (Voice) : {settings.audio_model} [Quantized]")
    print("─" * 70)
    print("  API Endpoints on Port 8080:")
    print(f"    • Chat (SSE)     : POST http://{host}:{port}/v1/chat/completions")
    print(f"    • Images         : POST http://{host}:{port}/v1/images/generations")
    print(f"    • Audio Speech   : POST http://{host}:{port}/v1/audio/speech")
    print(f"    • Realtime Voice : POST http://{host}:{port}/v1/audio/duplex_stream")
    print(f"    • Catalog        : GET  http://{host}:{port}/v1/models")
    print(f"    • VRAM Stats     : GET  http://{host}:{port}/v1/system/vram")
    print(f"    • VRAM Evict     : POST http://{host}:{port}/v1/system/unload")
    print("═" * 70 + "\n")


app = create_app()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Model Infrastructure Server on Apple Silicon"
    )
    parser.add_argument("--host", default=settings.host, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=settings.port, help="Bind port (default: 8080)")
    parser.add_argument("--text-model", default=None, help="Override text model ID")
    parser.add_argument("--image-model", default=None, help="Override image model ID")
    parser.add_argument("--audio-model", default=None, help="Override audio model ID")
    parser.add_argument("--prefer-bf16", action="store_true", help="Force unquantized BF16 for text")
    parser.add_argument("--idle-ttl", type=int, default=settings.model_idle_ttl_sec, help="VRAM idle eviction TTL (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    settings.host = args.host
    settings.port = args.port
    if args.text_model:
        settings.text_model = args.text_model
    if args.image_model:
        settings.image_model = args.image_model
    if args.audio_model:
        settings.audio_model = args.audio_model
    if args.prefer_bf16:
        settings.prefer_bf16_text = True
    settings.model_idle_ttl_sec = args.idle_ttl

    print_banner(settings.host, settings.port)
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
