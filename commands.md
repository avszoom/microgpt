# MicroGPT — Commands

All commands are run from the `microgpt/` project root.

## 1. Setup (one time)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the dataset

```bash
python scripts/download_data.py                    # wikitext-2 (~12MB, default)
python scripts/download_data.py wikitext-103-raw-v1  # larger (~500MB)
```

Writes the corpus to `data/wiki.txt`.

## 3. Train

```bash
# quick smoke test (tiny model, ~1 min)
python -m microgpt.train --steps 50 --vocab-size 512 --n-layer 2 --n-head 2 --n-embd 128 --block-size 64 --batch-size 16

# full run with default config (~14M params)
python -m microgpt.train

# scale up
python -m microgpt.train --n-layer 8 --n-embd 512 --block-size 384 --steps 10000

# force a specific device
python -m microgpt.train --device cpu      # or: mps, cuda
```

Outputs:
- `checkpoints/tokenizer.json` — trained BPE tokenizer (reused across runs)
- `checkpoints/ckpt.pt` — best model checkpoint (by val loss)
- `outputs/loss_curve.png` — training/validation loss plot

> Retraining the tokenizer: delete `checkpoints/tokenizer.json` (e.g. after changing the
> dataset or `--vocab-size`). Delete `checkpoints/ckpt.pt` to start the model from scratch.

## 4. Generate text

```bash
python -m microgpt.generate --prompt "The history of" --max-new-tokens 300 --temperature 0.8 --top-k 50
```

Flags: `--prompt`, `--max-new-tokens`, `--temperature` (higher = more random),
`--top-k` (limit sampling to top-k tokens), `--device`.

## 5. Run the web demo

```bash
streamlit run app/streamlit_app.py
```

## 6. Run the tokenizer self-test

```bash
python microgpt/tokenizer.py
```

## Train in the background (long runs)

```bash
nohup python -u -m microgpt.train > outputs/train.log 2>&1 &
tail -f outputs/train.log
```
