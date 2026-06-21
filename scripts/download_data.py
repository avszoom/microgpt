"""Download a Wikipedia-derived text corpus (wikitext) into data/wiki.txt.

Primary path uses the Hugging Face `datasets` library (wikitext-103-raw). If
that is unavailable, falls back to streaming the smaller wikitext-2-raw archive
directly from the public mirror.
"""

from __future__ import annotations

import io
import logging
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "wiki.txt"

WIKITEXT2_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    "wikitext-2-raw-v1.zip"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_data")


def _via_datasets(config: str) -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("`datasets` not installed; using direct-download fallback.")
        return None
    try:
        logger.info("Loading wikitext/%s via Hugging Face datasets ...", config)
        ds = load_dataset("wikitext", config, split="train")
        text = "\n".join(line for line in ds["text"] if line.strip())
        return text
    except Exception as exc:  # network / hub issues -> fallback
        logger.warning("datasets path failed (%s); falling back to direct download.", exc)
        return None


def _via_direct() -> str:
    logger.info("Downloading %s ...", WIKITEXT2_URL)
    resp = requests.get(WIKITEXT2_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("wiki.train.raw"))
        with zf.open(name) as fh:
            text = fh.read().decode("utf-8")
    return "\n".join(line for line in text.splitlines() if line.strip())


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = sys.argv[1] if len(sys.argv) > 1 else "wikitext-103-raw-v1"

    text = _via_datasets(config)
    if not text:
        text = _via_direct()

    OUT_PATH.write_text(text, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.info("Wrote %s (%.1f MB, %d chars)", OUT_PATH, size_mb, len(text))
    if size_mb < 0.01:
        logger.error("Corpus looks empty; aborting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
