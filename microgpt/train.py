"""Train MicroGPT from scratch.

Usage:
    python -m microgpt.train                 # full run with config defaults
    python -m microgpt.train --steps 200     # quick smoke test
    python -m microgpt.train --n-layer 4 --n-embd 256 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import torch

from .config import TrainConfig
from .dataset import TextDataset
from .device import get_device
from .model import GPT
from .tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MicroGPT")
    p.add_argument("--steps", type=int, default=None, help="override max training steps")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda | mps | cpu")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.steps is not None:
        cfg.max_steps = args.steps
        cfg.eval_interval = min(cfg.eval_interval, max(1, args.steps // 4))
        cfg.warmup_steps = min(cfg.warmup_steps, max(1, args.steps // 10))
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.block_size is not None:
        cfg.model.block_size = args.block_size
    if args.n_layer is not None:
        cfg.model.n_layer = args.n_layer
    if args.n_head is not None:
        cfg.model.n_head = args.n_head
    if args.n_embd is not None:
        cfg.model.n_embd = args.n_embd
    cfg.model.vocab_size = cfg.vocab_size
    return cfg


def lr_for_step(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, data: TextDataset, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = data.get_batch(split, cfg.batch_size)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_loss_curve(history: list[dict], path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping loss curve.")
        return
    steps = [h["step"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [h["train"] for h in history], label="train")
    plt.plot(steps, [h["val"] for h in history], label="val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("MicroGPT training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curve -> %s", path)


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    device = get_device(args.device)

    if not cfg.data_path.exists():
        logger.error("Corpus not found at %s. Run: python scripts/download_data.py", cfg.data_path)
        return 1

    # --- tokenizer: train once, then reuse ---
    if cfg.tokenizer_path.exists():
        tokenizer = BPETokenizer.load(cfg.tokenizer_path)
    else:
        logger.info("Training BPE tokenizer (vocab_size=%d) ...", cfg.vocab_size)
        text = cfg.data_path.read_text(encoding="utf-8")
        tokenizer = BPETokenizer().train(text, vocab_size=cfg.vocab_size)
        tokenizer.save(cfg.tokenizer_path)
    # keep model vocab in sync with the actual trained tokenizer
    cfg.model.vocab_size = tokenizer.vocab_size

    data = TextDataset(cfg.data_path, tokenizer, cfg.model.block_size, cfg.val_split, device)

    model = GPT(cfg.model).to(device)
    logger.info("Model parameters: %.2fM", model.num_params() / 1e6)
    optimizer = model.configure_optimizer(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2)
    )

    history: list[dict] = []
    best_val = float("inf")
    model.train()
    t0 = time.time()

    for step in range(cfg.max_steps + 1):
        lr = lr_for_step(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, data, cfg)
            history.append({"step": step, **losses})
            logger.info(
                "step %d | train %.4f | val %.4f | lr %.2e | %.1fs",
                step, losses["train"], losses["val"], lr, time.time() - t0,
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": cfg.model.as_dict(),
                        "step": step,
                        "val_loss": best_val,
                    },
                    cfg.ckpt_path,
                )
                logger.info("  saved checkpoint (val %.4f) -> %s", best_val, cfg.ckpt_path)

        if step == cfg.max_steps:
            break

        x, y = data.get_batch("train", cfg.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            logger.info("  step %d | batch loss %.4f", step, loss.item())

    save_loss_curve(history, cfg.loss_curve_path)

    # quick generation sample so the run ends with something readable
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, max_new_tokens=cfg.sample_tokens, temperature=0.8, top_k=50)
    logger.info("--- sample ---\n%s\n--------------", tokenizer.decode(sample[0].tolist()))
    logger.info("Done. Best val loss: %.4f", best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
