"""Configuration settings for local-model-infra."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # Server network bindings
    host: str = Field(default_factory=lambda: os.getenv("MODEL_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("MODEL_PORT", "8080")))

    # Base paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    cache_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MODELS_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
        )
    )

    # Model identifiers
    # Text Model: Spark-X2.5-4B (1M context hybrid attention)
    text_model: str = Field(
        default_factory=lambda: os.getenv(
            "LOCAL_TEXT_MODEL", "hermitdave/Spark-X2.5-4B-MLX-4bit"
        )
    )
    text_model_bf16: str = Field(
        default_factory=lambda: os.getenv("LOCAL_TEXT_MODEL_BF16", "XHToken/Spark-X2.5-4B")
    )
    prefer_bf16_text: bool = Field(
        default_factory=lambda: os.getenv("PREFER_BF16_TEXT", "false").lower() in ("true", "1", "yes")
    )

    # Image Model: Qwen-Image-2.1 (7B DiT + Text Encoder)
    image_model: str = Field(
        default_factory=lambda: os.getenv(
            "LOCAL_IMAGE_MODEL", "mlx-community/Qwen-Image-2.1-MLX-4bit"
        )
    )

    # Audio Model: StepAudio 3 Realtime / Step-Audio-TTS-3B
    audio_model: str = Field(
        default_factory=lambda: os.getenv(
            "LOCAL_AUDIO_MODEL", "stepfun-ai/Step-Audio-TTS-3B"
        )
    )

    # Hardware & Memory Protection
    model_idle_ttl_sec: int = Field(
        default_factory=lambda: int(os.getenv("MODEL_IDLE_TTL_SECONDS", "300"))
    )
    max_wired_memory_gb: float = Field(
        default_factory=lambda: float(os.getenv("MAX_WIRED_MEMORY_GB", "11.8"))
    )
    enable_simulation_fallback: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SIMULATION_FALLBACK", "true").lower() in ("true", "1", "yes")
    )


settings = Settings()
