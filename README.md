# MicroGPT

A tiny GPT-style transformer language model trained **from scratch** in PyTorch — no pretrained weights, no Hugging Face models. It learns to generate text one token at a time by predicting the next token.

This is **Project 1** of the [LLM Systems Lab](https://github.com/avszoom) roadmap: *Train → Serve → Optimize*.

## What it does

MicroGPT implements every core piece of a modern LLM by hand:

- **BPE tokenizer** (`microgpt/tokenizer.py`) — byte-level Byte-Pair Encoding, trained from scratch
- **Dataset loader** (`microgpt/dataset.py`) — chunks a corpus into `(x, y)` next-token windows
- **Decoder-only transformer** (`microgpt/model.py`) — token + positional embeddings → stacked blocks (causal multi-head self-attention + MLP) → next-token head, with weight tying
- **Training loop** (`microgpt/train.py`) — cross-entropy loss, AdamW, cosine LR schedule with warmup, periodic eval, checkpointing, loss-curve plot
- **Generation** (`microgpt/generate.py`) — autoregressive sampling with temperature and top-k
- **Streamlit demo** (`app/streamlit_app.py`) — type a prompt, tune sampling, view output + loss curve

## Project structure

```
microgpt/
├── microgpt/                # core library
│   ├── config.py            # GPTConfig + TrainConfig (all hyperparameters)
│   ├── device.py            # auto-pick CUDA -> MPS -> CPU
│   ├── tokenizer.py         # from-scratch byte-level BPE
│   ├── dataset.py           # token batches
│   ├── model.py             # the GPT
│   ├── train.py             # training loop (CLI)
│   └── generate.py          # sampling (CLI)
├── app/streamlit_app.py     # web demo
├── scripts/download_data.py # fetch wikitext -> data/wiki.txt
├── configs/default.yaml     # documented default hyperparameters
├── data/                    # corpus (gitignored)
├── checkpoints/             # weights + tokenizer (gitignored)
├── outputs/                 # loss curve, samples (gitignored)
└── requirements.txt
```

## Setup

```bash
cd microgpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Download the corpus** (Wikipedia subset — wikitext):

```bash
python scripts/download_data.py            # wikitext-103-raw (default)
python scripts/download_data.py wikitext-2-raw-v1   # smaller/faster
```

**2. Train** (trains the tokenizer on first run, then the model):

```bash
python -m microgpt.train --steps 200       # quick smoke test
python -m microgpt.train                    # full run (config defaults)
python -m microgpt.train --n-layer 8 --n-embd 512 --block-size 384   # scale up
```

Checkpoints land in `checkpoints/ckpt.pt`; the loss curve is saved to `outputs/loss_curve.png`.

**3. Generate** text from a prompt:

```bash
python -m microgpt.generate --prompt "The history of" --max-new-tokens 300 --temperature 0.8 --top-k 50
```

**4. Run the demo:**

```bash
streamlit run app/streamlit_app.py
```

## Configuration

All hyperparameters live in `microgpt/config.py` (mirrored, documented, in `configs/default.yaml`). The default model is **~10–15M params** (`6 layers, 6 heads, 384 dim, 256 context`) — small enough to train on a local GPU (CUDA or Apple Silicon MPS) in a reasonable time, and a real transformer. Scale it up by raising `n_layer` / `n_embd` / `block_size`.

The device is auto-detected: **CUDA → MPS → CPU**. Force one with `--device cpu`.

## What you'll learn

Tokenization (BPE) · embeddings · self-attention · positional encoding · the pretraining objective · cross-entropy loss & perplexity · optimizer/LR scheduling · why training is expensive.

## License

MIT
