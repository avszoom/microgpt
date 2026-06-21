"""Byte-level Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

This is a learning-oriented implementation in the spirit of GPT-2's BPE:

1. Text is split into chunks with a regex (so merges never cross obvious
   word/punctuation boundaries).
2. Each chunk is encoded as raw UTF-8 bytes (base vocabulary = 256 byte tokens).
3. Training repeatedly merges the most frequent adjacent token pair until the
   target vocabulary size is reached. Merges are frequency-weighted over the set
   of *unique* chunks, which keeps training fast on multi-MB corpora.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# GPT-2 style pre-tokenization: keep contractions, words, numbers, punctuation
# and leading spaces as separate chunks. The Unicode-aware \p{...} pattern needs
# the third-party `regex` module; stdlib `re` falls back to an ASCII-class pattern.
try:  # pragma: no cover - environment dependent
    import regex as _regex

    _SPLIT_PATTERN = _regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""",
        _regex.UNICODE,
    )
except ImportError:  # pragma: no cover
    _SPLIT_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""",
        re.UNICODE,
    )

_findall = _SPLIT_PATTERN.findall


def _get_pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter:
    """Count adjacent id pairs across all chunks, weighted by chunk frequency."""
    counts: Counter = Counter()
    for ids, freq in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """A trainable byte-level BPE tokenizer."""

    def __init__(self) -> None:
        # merges: ordered mapping of (id, id) -> new_id (rank == insertion order)
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> raw bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ train
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")

        num_merges = vocab_size - 256
        word_freqs: Counter = Counter(_findall(text))
        chunks: dict[tuple[int, ...], int] = {
            tuple(word.encode("utf-8")): freq for word, freq in word_freqs.items()
        }

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for step in range(num_merges):
            pair_counts = _get_pair_counts(chunks)
            if not pair_counts:
                logger.info("No more pairs to merge; stopping at vocab_size=%d", 256 + step)
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                logger.info("Most frequent pair occurs <2 times; stopping early")
                break

            new_id = 256 + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            chunks = {
                _merge(ids, best_pair, new_id): freq for ids, freq in chunks.items()
            }

            if verbose and (step + 1) % 500 == 0:
                logger.info("merge %d/%d -> id %d (count=%d)", step + 1, num_merges, new_id, best_count)

        logger.info("Trained BPE tokenizer: vocab_size=%d", len(self.vocab))
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---------------------------------------------------------------- encode
    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # find the mergeable pair with the lowest rank (earliest learned)
            pairs = set(zip(ids, ids[1:]))
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = list(_merge(tuple(ids), candidate, self.merges[candidate]))
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    # ---------------------------------------------------------------- decode
    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [[p0, p1, nid] for (p0, p1), nid in self.merges.items()],
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Saved tokenizer -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = {}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        # rebuild merges in saved order so vocab bytes resolve correctly
        for p0, p1, nid in sorted(payload["merges"], key=lambda r: r[2]):
            tok.merges[(p0, p1)] = nid
            tok.vocab[nid] = tok.vocab[p0] + tok.vocab[p1]
        logger.info("Loaded tokenizer (vocab_size=%d) from %s", tok.vocab_size, path)
        return tok


def _self_test() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sample = (
        "The history of natural language processing began in the 1950s. "
        "Machine learning models learn patterns from data. " * 50
    )
    tok = BPETokenizer().train(sample, vocab_size=400)
    text = "The history of language models."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"text   : {text!r}")
    print(f"ids    : {ids}")
    print(f"decoded: {decoded!r}")
    assert decoded == text, "round-trip failed"
    print("OK: encode/decode round-trip passed.")


if __name__ == "__main__":
    _self_test()
