"""Device selection helper: prefer local GPU (CUDA, then Apple MPS), else CPU."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Order of preference: explicit `prefer` -> cuda -> mps -> cpu.
    """
    if prefer:
        logger.info("Using user-requested device: %s", prefer)
        return torch.device(prefer)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA available -> using GPU: %s", name)
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        logger.info("Apple MPS available -> using GPU (Metal)")
        return torch.device("mps")

    logger.info("No GPU found -> using CPU")
    return torch.device("cpu")
