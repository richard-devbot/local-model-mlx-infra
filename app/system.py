"""System and Apple Metal hardware telemetry utilities."""

from __future__ import annotations

import os
import shutil
from typing import Any
import psutil

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    MLX_AVAILABLE = False


def get_metal_stats() -> dict[str, Any]:
    """Retrieve Apple Silicon Metal GPU memory stats and limits."""
    if not MLX_AVAILABLE or mx is None:
        return {"available": False, "reason": "mlx not installed"}

    try:
        info = mx.device_info()
        return {
            "available": True,
            "device_name": info.get("device_name", "Apple Silicon"),
            "architecture": info.get("architecture", "Unknown"),
            "memory_size_gb": round(info.get("memory_size", 0) / (1024**3), 2),
            "max_working_set_gb": round(
                info.get("max_recommended_working_set_size", 0) / (1024**3), 2
            ),
            "metal_active": mx.metal.is_available(),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def get_system_memory() -> dict[str, Any]:
    """Retrieve host system RAM usage via psutil."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    proc = psutil.Process(os.getpid())
    return {
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_percent": vm.percent,
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_percent": swap.percent,
        "process_rss_gb": round(proc.memory_info().rss / (1024**3), 3),
    }


def get_disk_space(path: str = "/") -> dict[str, Any]:
    """Retrieve host disk storage metrics in GB."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "path": path,
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_full_telemetry(active_model: str | None = None) -> dict[str, Any]:
    """Combined telemetry payload for production monitoring."""
    return {
        "status": "healthy",
        "active_model": active_model,
        "metal": get_metal_stats(),
        "memory": get_system_memory(),
        "disk": get_disk_space(),
    }
