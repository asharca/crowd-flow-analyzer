"""Centralized device and thread configuration for the ML pipeline."""

import logging
import os

import torch

from app.config import settings

logger = logging.getLogger(__name__)


def resolve_device() -> str:
    """Return the best available torch device string."""
    if settings.ml_device != "auto":
        return settings.ml_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_demographics_workers() -> int:
    """Return number of parallel workers for demographics analysis."""
    if settings.demographics_workers > 0:
        return settings.demographics_workers
    cpu_count = os.cpu_count() or 4
    return max(2, cpu_count - 1)


def _cuda_device_index(device: str) -> int:
    """Parse device string like 'cuda', 'cuda:0', 'cuda:1' → int index."""
    if ":" in device:
        return int(device.split(":")[1])
    return 0


def resolve_yolo_batch_size(device: str) -> int:
    """Return YOLO inference batch size based on available device memory."""
    if settings.yolo_batch_size > 0:
        return settings.yolo_batch_size
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = _cuda_device_index(device)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        if vram_gb >= 16:
            return 128
        if vram_gb >= 8:
            return 64
        if vram_gb >= 4:
            return 32
        return 16
    # CPU: scale with core count but keep memory reasonable
    cpu_count = os.cpu_count() or 4
    return min(16, cpu_count * 2)


def resolve_frame_skip(device: str) -> int:
    """Return frame skip based on device — process every frame on GPU."""
    if settings.frame_skip > 0:
        return settings.frame_skip
    return 1 if device.startswith("cuda") else 3


def resolve_yolo_model(device: str) -> str:
    """Return YOLO model — use larger model on GPU for better accuracy + utilization."""
    if settings.yolo_model:
        return settings.yolo_model
    return "yolo11m.pt" if device.startswith("cuda") else "yolo11n.pt"


def resolve_mivolo_batch_size(device: str) -> int:
    """Return MiVOLO batch size based on VRAM."""
    if settings.mivolo_batch_size > 0:
        return settings.mivolo_batch_size
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = _cuda_device_index(device)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        if vram_gb >= 16:
            return 128
        if vram_gb >= 8:
            return 64
        return 32
    return 16


def configure_threads() -> None:
    """Set CPU thread counts for torch, OpenCV, and NumPy."""
    import cv2
    num_threads = settings.ml_num_threads or (os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    cv2.setNumThreads(num_threads)
    # Let OpenBLAS/MKL (used by NumPy) use all cores
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    device = resolve_device()
    if device.startswith("cuda"):
        idx = _cuda_device_index(device)
        gpu_name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        logger.info("GPU enabled: %s (%.1f GB VRAM), device=%s", gpu_name, vram_gb, device)
    else:
        logger.info("GPU not available, using %s", device)
    logger.info("Thread config: %d threads", num_threads)


# Run once at import time so all ML modules benefit
configure_threads()

DEVICE: str = resolve_device()
DEMOGRAPHICS_WORKERS: int = resolve_demographics_workers()
YOLO_BATCH_SIZE: int = resolve_yolo_batch_size(DEVICE)
FRAME_SKIP: int = resolve_frame_skip(DEVICE)
YOLO_MODEL: str = resolve_yolo_model(DEVICE)
MIVOLO_BATCH_SIZE: int = resolve_mivolo_batch_size(DEVICE)

logger.info(
    "ML config: device=%s, yolo=%s, batch=%d, frame_skip=%d, mivolo_batch=%d",
    DEVICE, YOLO_MODEL, YOLO_BATCH_SIZE, FRAME_SKIP, MIVOLO_BATCH_SIZE,
)
