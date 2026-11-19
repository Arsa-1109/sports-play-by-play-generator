"""Core implementation of Sports Play-by-Play Generator (GPT from scratch)."""
from __future__ import annotations

import math
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.corpus import SPORTS_ARTICLES

ARTICLE_BOUNDARY = "<|article|>"
DEFAULT_CHECKPOINT_PATH = Path("data/sports_gpt.pt")

# Regex partitioning any Unicode string into contiguous chunks for fast BPE training & encoding.
_BPE_SPLIT_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


# ---------------------------------------------------------------------------
# Milestone 1: Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize line endings and collapse redundant horizontal whitespace."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def load_corpus(corpus_path: str | Path | None = None) -> list[str]:
    """Load and normalize sports articles from a file path or the built-in corpus."""
    if corpus_path is None:
        return [normalize_text(article) for article in SPORTS_ARTICLES if normalize_text(article)]

    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if ARTICLE_BOUNDARY in raw_text:
        chunks = raw_text.split(ARTICLE_BOUNDARY)
    elif "\n---\n" in raw_text:
        chunks = raw_text.split("\n---\n")
    else:
        chunks = [raw_text]

    articles = [normalize_text(chunk) for chunk in chunks if normalize_text(chunk)]
    if not articles:
        raise ValueError(f"Corpus file contains no usable text: {path}")
    return articles


def prepare_corpus_text(articles: list[str], boundary_token: str = ARTICLE_BOUNDARY) -> str:
    """Join normalized articles using explicit boundary tokens."""
    cleaned = [normalize_text(a) for a in articles if normalize_text(a)]
    if not cleaned:
        raise ValueError("Cannot prepare corpus text from an empty article list.")
    return f"\n{boundary_token}\n".join(cleaned)


def train_val_split(
    token_ids: list[int] | torch.Tensor,
    val_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split a token sequence into train and validation 1D tensors."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    data = token_ids.to(dtype=torch.long) if isinstance(token_ids, torch.Tensor) else torch.tensor(token_ids, dtype=torch.long)
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token sequence, got shape {tuple(data.shape)}")
    if data.numel() < 2:
        raise ValueError("Need at least 2 tokens to split into train and validation sets.")

    val_len = max(1, int(data.numel() * val_ratio))
    split_idx = max(1, data.numel() - val_len)
    return data[:split_idx], data[split_idx:]


def milestone_1(corpus_path: str | Path | None = None) -> dict[str, Any]:
    """Milestone 1 entrypoint: load, normalize, and inspect the sports corpus."""
    articles = load_corpus(corpus_path)
    text = prepare_corpus_text(articles)
    return {
        "articles": articles,
        "num_articles": len(articles),
        "total_chars": len(text),
        "text": text,
        "sample_preview": text[:240],
    }


# --- Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
# Hint: Byte-Pair Tokenizer; Token Embeddings
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal Self-Attention; Transformer Block
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Train the GPT model using the sports text data.
# Hint: Training Loop; Loss Calculation
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Develop a text generation function to produce short commentary snippets.
# Hint: Sampling; Greedy Decoding
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
# Hint: User Input Handling
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
