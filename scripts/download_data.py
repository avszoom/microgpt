"""Download a Wikipedia-derived text corpus (wikitext) into data/wiki.txt.

Primary path uses the Hugging Face `datasets` library against the canonical
`Salesforce/wikitext` parquet repo. If `datasets` is unavailable, falls back to
downloading the parquet shards directly from the Hub and reading them with
pyarrow.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "wiki.txt"

HF_REPO = "Salesforce/wikitext"
# directory listing API for the parquet shards of a given config
HF_TREE_URL = "https://huggingface.co/api/datasets/Salesforce/wikitext/tree/main/{config}"
HF_RESOLVE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/{path}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_data")


def _join(rows) -> str:
    return "\n".join(line for line in rows if line and line.strip())


def _via_datasets(config: str) -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("`datasets` not installed; using direct-download fallback.")
        return None
    try:
        logger.info("Loading %s/%s via Hugging Face datasets ...", HF_REPO, config)
        ds = load_dataset(HF_REPO, config, split="train")
        return _join(ds["text"])
    except Exception as exc:  # network / hub issues -> fallback
        logger.warning("datasets path failed (%s); falling back to direct download.", exc)
        return None


def _via_direct(config: str) -> str:
    import pyarrow.parquet as pq

    logger.info("Listing parquet shards for %s ...", config)
    tree = requests.get(HF_TREE_URL.format(config=config), timeout=60).json()
    shards = sorted(
        e["path"] for e in tree
        if e.get("path", "").endswith(".parquet") and "train" in e["path"]
    )
    if not shards:
        raise RuntimeError(f"No train parquet shards found for config {config!r}")

    rows: list[str] = []
    for path in shards:
        logger.info("Downloading %s ...", path)
        resp = requests.get(HF_RESOLVE_URL.format(path=path), timeout=300)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        rows.extend(table.column("text").to_pylist())
    return _join(rows)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # wikitext-2 (~12MB) is the right size for a mini GPT on a local GPU.
    # Pass "wikitext-103-raw-v1" as an arg for a much larger (~500MB) corpus.
    config = sys.argv[1] if len(sys.argv) > 1 else "wikitext-2-raw-v1"

    text = _via_datasets(config)
    if not text:
        text = _via_direct(config)

    OUT_PATH.write_text(text, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.info("Wrote %s (%.1f MB, %d chars)", OUT_PATH, size_mb, len(text))
    if size_mb < 0.01:
        logger.error("Corpus looks empty; aborting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
