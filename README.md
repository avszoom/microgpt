# MicroGPT

A tiny GPT-style transformer language model trained **from scratch** in PyTorch — no pretrained weights, no Hugging Face models. It learns to generate text one character at a time by predicting the next character.

This is **Project 1** of the [LLM Systems Lab](https://github.com/avszoom) roadmap: *Train → Serve → Optimize*.

## What it does

MicroGPT implements every core piece of a modern LLM by hand:

- **Char-level tokenizer** — maps each character ↔ integer
- **Dataset loader** — chunks a text corpus into input/target sequences
- **Decoder-only transformer** — token + positional embeddings → stacked blocks (multi-head self-attention + MLP) → next-token head
- **Training loop** — cross-entropy loss, AdamW, periodic eval, train/val loss logging
- **Generation** — autoregressive sampling with temperature and top-k
- **Visualization** — loss curve plot + before/after generated samples

## Why

Most people only call APIs or fine-tune. Building a GPT from scratch shows you understand the actual foundations: tokenization, embeddings, self-attention, positional encoding, the pretraining objective, cross-entropy loss, and why training is expensive.

## Project structure

```
microgpt/
├── microgpt/      # core library (model, tokenizer, dataset, train, generate)
├── scripts/       # helpers (e.g. download dataset)
├── configs/       # training hyperparameter configs
├── data/          # raw text corpus
├── checkpoints/   # saved model weights
├── outputs/       # loss curves, generated samples
└── requirements.txt
```

## Tech stack

Python · PyTorch · NumPy · Matplotlib

## Status

🚧 Work in progress — scaffolding complete, model implementation next.

## License

MIT
