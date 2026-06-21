"""Central hyperparameter definitions for MicroGPT."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class GPTConfig:
    """Architecture hyperparameters for the GPT model."""

    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    """Training-loop hyperparameters and filesystem paths."""

    # data / tokenizer
    data_path: Path = DATA_DIR / "wiki.txt"
    tokenizer_path: Path = CKPT_DIR / "tokenizer.json"
    ckpt_path: Path = CKPT_DIR / "ckpt.pt"
    loss_curve_path: Path = OUTPUT_DIR / "loss_curve.png"
    vocab_size: int = 8192
    val_split: float = 0.1

    # optimization
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # evaluation / logging
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 50
    sample_tokens: int = 200

    seed: int = 1337

    model: GPTConfig = field(default_factory=GPTConfig)

    def __post_init__(self) -> None:
        # keep vocab_size consistent across train + model config
        self.model.vocab_size = self.vocab_size
