"""Generate text from a trained MicroGPT checkpoint.

Usage:
    python -m microgpt.generate --prompt "The history of" --max-new-tokens 300
"""

from __future__ import annotations

import argparse
import logging

import torch

from .config import GPTConfig, TrainConfig
from .device import get_device
from .model import GPT
from .tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with MicroGPT")
    p.add_argument("--prompt", type=str, default="\n", help="seed text")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(device: torch.device) -> tuple[GPT, BPETokenizer]:
    cfg = TrainConfig()
    if not cfg.ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {cfg.ckpt_path}. Train first: python -m microgpt.train"
        )
    if not cfg.tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer at {cfg.tokenizer_path}.")

    tokenizer = BPETokenizer.load(cfg.tokenizer_path)
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    logger.info(
        "Loaded checkpoint (step %s, val_loss %.4f)", ckpt.get("step"), ckpt.get("val_loss", float("nan"))
    )
    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    model, tokenizer = load_model(device)
    text = generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.temperature, args.top_k, device,
    )
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
