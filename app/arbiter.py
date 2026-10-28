"""VRAM Arbiter: Manages single-model lifecycle, memory swapping, and auto-eviction."""

from __future__ import annotations

import asyncio
import logging
import time
import contextlib
from typing import TYPE_CHECKING, Callable, Dict

from app.config import settings
from app.system import get_system_memory

if TYPE_CHECKING:
    from app.engines.base import BaseModelEngine

logger = logging.getLogger("local-infra.arbiter")


class VRAMArbiter:
    """Coordinates GPU/VRAM allocation on Apple Silicon so only one heavy model

    is loaded at any given time, preventing memory exhaustion on 16GB systems.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_engine: BaseModelEngine | None = None
        self._last_access_time: float = time.time()
        self._engine_factories: Dict[str, Callable[[str], BaseModelEngine]] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._running = False
        self._in_flight = 0

    def register_factory(
        self, modality: str, factory: Callable[[str], BaseModelEngine]
    ) -> None:
        """Register a factory callable that instantiates an engine for a modality."""
        self._engine_factories[modality] = factory

    @property
    def active_model_id(self) -> str | None:
        """Return ID of currently active model in memory, if any."""
        return self._active_engine.model_id if self._active_engine else None

    @property
    def active_modality(self) -> str | None:
        """Return modality of currently active model in memory."""
        return self._active_engine.modality if self._active_engine else None

    def touch(self) -> None:
        """Update last access time to prevent idle timeout."""
        self._last_access_time = time.time()

    @contextlib.asynccontextmanager
    async def track_activity(self):
        """Context manager to track in-flight requests and prevent eviction."""
        self._in_flight += 1
        self._last_access_time = time.time()
        try:
            yield
        finally:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_access_time = time.time()

    async def acquire_engine(
        self, modality: str, target_model_id: str
    ) -> BaseModelEngine:
        """Safely acquire an engine for inference.

        If the active engine does not match, the previous model is evicted,
        caches are cleared, and the requested model is loaded.
        """
        async with self._lock:
            self._last_access_time = time.time()

            # Warm path: already loaded and matches model ID
            if (
                self._active_engine is not None
                and self._active_engine.is_loaded
                and self._active_engine.model_id == target_model_id
            ):
                logger.debug(f"Warm model hit: {target_model_id}")
                return self._active_engine

            # Swap path: unload previous model if present
            if self._active_engine is not None:
                logger.info(
                    f"Model swap required: Unloading '{self._active_engine.model_id}' "
                    f"to make room for '{target_model_id}'"
                )
                try:
                    await asyncio.to_thread(self._active_engine.unload)
                except Exception as exc:
                    logger.error(f"Error unloading previous model: {exc}")
                self._active_engine = None

            # Create and load new target engine
            factory = self._engine_factories.get(modality)
            if factory is None:
                raise ValueError(f"No engine factory registered for modality '{modality}'")

            new_engine = factory(target_model_id)

            logger.info(
                f"Loading {modality} model '{target_model_id}' into Unified Memory "
                f"(est. {new_engine.estimate_vram_gb()} GB)..."
            )

            # Check available host RAM before loading
            mem_info = get_system_memory()
            available_gb = mem_info.get("ram_available_gb", 0.0)
            logger.info(f"Host available RAM before load: {available_gb} GB")

            # Run loading in worker thread to prevent blocking FastAPI event loop
            await asyncio.to_thread(new_engine.load)

            self._active_engine = new_engine
            self._last_access_time = time.time()
            logger.info(f"Model '{target_model_id}' successfully loaded and ready.")
            return self._active_engine

    async def unload_active(self) -> None:
        """Manually unload currently active model and reclaim memory."""
        async with self._lock:
            if self._active_engine is not None:
                logger.info(f"Manual unload requested for '{self._active_engine.model_id}'")
                await asyncio.to_thread(self._active_engine.unload)
                self._active_engine = None

    def start_idle_monitor(self) -> None:
        """Start the background idle eviction monitor."""
        if self._monitor_task is None or self._monitor_task.done():
            self._running = True
            self._monitor_task = asyncio.create_task(self._idle_monitor_loop())
            logger.info(
                f"Started idle VRAM monitor (TTL: {settings.model_idle_ttl_sec}s)"
            )

    async def stop_idle_monitor(self) -> None:
        """Stop idle eviction monitor."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _idle_monitor_loop(self) -> None:
        """Background coroutine that unloads models when idle exceeds TTL."""
        while self._running:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds
                if settings.model_idle_ttl_sec <= 0:
                    continue
                if self._in_flight > 0:
                    self._last_access_time = time.time()
                    continue
                if self._active_engine is not None and self._active_engine.is_loaded:
                    idle_duration = time.time() - self._last_access_time
                    if idle_duration > settings.model_idle_ttl_sec:
                        logger.info(
                            f"Model '{self._active_engine.model_id}' idle for {int(idle_duration)}s "
                            f"(TTL: {settings.model_idle_ttl_sec}s). Auto-evicting from VRAM..."
                        )
                        async with self._lock:
                            if self._active_engine is not None:
                                await asyncio.to_thread(self._active_engine.unload)
                                self._active_engine = None
                                logger.info("Auto-eviction complete. VRAM reclaimed.")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in idle monitor loop: {exc}")


# Global arbiter instance
arbiter = VRAMArbiter()
