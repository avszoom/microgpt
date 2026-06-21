"""Token dataset: encode a corpus once, then serve random training batches."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class TextDataset:
    """Holds the full corpus as a 1-D tensor of token ids and yields batches.

    Each batch is a contiguous-window language-modeling sample:
        x = tokens[i : i + block_size]
        y = tokens[i + 1 : i + block_size + 1]   (x shifted left by one)
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: BPETokenizer,
        block_size: int,
        val_split: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        self.block_size = block_size
        self.device = torch.device(device)

        text = Path(data_path).read_text(encoding="utf-8")
        logger.info("Loaded corpus: %d characters", len(text))

        ids = tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)
        logger.info("Encoded corpus: %d tokens", data.numel())

        if data.numel() <= block_size + 1:
            raise ValueError("Corpus is too small for the chosen block_size")

        n_val = int(data.numel() * val_split)
        self.train_data = data[:-n_val] if n_val > 0 else data
        self.val_data = data[-n_val:] if n_val > 0 else data
        logger.info(
            "Split tokens: train=%d val=%d", self.train_data.numel(), self.val_data.numel()
        )

    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        max_start = data.numel() - self.block_size - 1
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        # pin + non_blocking helps a little when copying to a CUDA device
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
