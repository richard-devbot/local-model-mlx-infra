"""Base interface for all modality model engines."""

from __future__ import annotations

import abc
import gc
import logging
from typing import Any

logger = logging.getLogger("local-infra.engine")


class BaseModelEngine(abc.ABC):
    """Abstract base class for locally loaded models."""

    def __init__(self, model_id: str, modality: str) -> None:
        self.model_id = model_id
        self.modality = modality
        self.is_loaded = False
        self._raw_model: Any = None
        self._raw_tokenizer: Any = None

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into host unified memory / Metal VRAM."""
        raise NotImplementedError

    def unload(self) -> None:
        """Unload model from memory, free Metal cache, and run garbage collection."""
        logger.info(f"Unloading {self.modality} model: {self.model_id}")
        self._raw_model = None
        self._raw_tokenizer = None
        self.is_loaded = False

        # Run Python garbage collection
        gc.collect()

        # Clear Apple Metal memory cache if MLX is loaded
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                mx.metal.clear_cache()
                logger.info("Cleared MLX Metal buffer cache.")
        except Exception as exc:
            logger.debug(f"MLX cache clear skipped: {exc}")

        # Clear PyTorch MPS cache if torch is loaded
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("Cleared PyTorch MPS cache.")
        except Exception:
            pass

    @abc.abstractmethod
    def estimate_vram_gb(self) -> float:
        """Estimated VRAM/RAM working set in GB."""
        raise NotImplementedError
