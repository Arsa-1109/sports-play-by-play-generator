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


# ---------------------------------------------------------------------------
# Milestone 2: Byte-Pair Tokenizer & Token/Positional Embeddings
# ---------------------------------------------------------------------------

def _merge_pair_in_sequence(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of `pair` in `ids` with `new_id`."""
    p0, p1 = pair
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == p0 and ids[i + 1] == p1:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level Byte-Pair Encoding (BPE) tokenizer built from scratch.

    Uses UTF-8 bytes (0..255) as the base vocabulary so any Unicode string
    encodes and decodes losslessly without out-of-vocabulary `<UNK>` tokens.
    """

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int = 512) -> "BPETokenizer":
        """Learn BPE merge rules from `text` up to `vocab_size` tokens."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256 for byte-level BPE, got {vocab_size}")

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        if not text:
            return self

        chunk_counts: Counter[tuple[int, ...]] = Counter()
        for chunk in _BPE_SPLIT_RE.findall(text):
            byte_seq = tuple(chunk.encode("utf-8"))
            if byte_seq:
                chunk_counts[byte_seq] += 1

        num_merges = vocab_size - 256
        for step in range(num_merges):
            pair_freqs: Counter[tuple[int, int]] = Counter()
            for seq, freq in chunk_counts.items():
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_freqs[(seq[i], seq[i + 1])] += freq

            if not pair_freqs:
                break

            # Deterministic tie-breaking: highest frequency, then lexicographically smallest pair.
            best_pair = min(pair_freqs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = 256 + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            updated_counts: Counter[tuple[int, ...]] = Counter()
            p0, p1 = best_pair
            for seq, freq in chunk_counts.items():
                if len(seq) >= 2 and p0 in seq and p1 in seq:
                    merged = tuple(_merge_pair_in_sequence(list(seq), best_pair, new_id))
                    updated_counts[merged] += freq
                else:
                    updated_counts[seq] += freq
            chunk_counts = updated_counts

        return self

    def encode(self, text: str) -> list[int]:
        """Encode arbitrary text into BPE token IDs."""
        if not text:
            return []

        output: list[int] = []
        cache: dict[str, list[int]] = {}
        for chunk in _BPE_SPLIT_RE.findall(text):
            cached = cache.get(chunk)
            if cached is not None:
                output.extend(cached)
                continue

            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                best_pair: tuple[int, int] | None = None
                best_rank = float("inf")
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    rank = self.merges.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                ids = _merge_pair_in_sequence(ids, best_pair, self.merges[best_pair])

            cache[chunk] = ids
            output.extend(ids)
        return output

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into a UTF-8 string."""
        if not ids:
            return ""
        raw_bytes = bytearray()
        for idx in ids:
            token_bytes = self.vocab.get(idx)
            if token_bytes is None:
                raise ValueError(f"Unknown token id: {idx}")
            raw_bytes.extend(token_bytes)
        return bytes(raw_bytes).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        """Serialize tokenizer merges and vocab size to JSON-compatible dict."""
        ordered_merges = sorted(self.merges.items(), key=lambda kv: kv[1])
        return {
            "vocab_size": self.vocab_size,
            "merges": [[p0, p1, new_id] for (p0, p1), new_id in ordered_merges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPETokenizer":
        """Restore a BPETokenizer from a serialized dictionary."""
        tok = cls()
        ordered_merges = sorted(data.get("merges", []), key=lambda item: item[2])
        for p0, p1, new_id in ordered_merges:
            tok.merges[(int(p0), int(p1))] = int(new_id)
            tok.vocab[int(new_id)] = tok.vocab[int(p0)] + tok.vocab[int(p1)]
        return tok


class TokenAndPositionalEmbedding(nn.Module):
    """Combines learned token embeddings with learned positional embeddings."""

    def __init__(self, vocab_size: int, block_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.block_size}")
        positions = torch.arange(seq_len, device=idx.device)
        tok = self.token_emb(idx)
        pos = self.pos_emb(positions).unsqueeze(0)
        return self.dropout(tok + pos)


def milestone_2(
    text: str | None = None,
    vocab_size: int = 400,
    d_model: int = 96,
    block_size: int = 64,
) -> dict[str, Any]:
    """Milestone 2 entrypoint: train BPE tokenizer and compute token + position embeddings."""
    if text is None:
        text = milestone_1()["text"]

    tokenizer = BPETokenizer().train(text, vocab_size=vocab_size)
    sample = "The striker fired a curling shot into the top corner!"
    sample_ids = tokenizer.encode(sample)
    emb_layer = TokenAndPositionalEmbedding(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=d_model,
    )
    idx_tensor = torch.tensor([sample_ids[:block_size]], dtype=torch.long)
    embedded = emb_layer(idx_tensor)
    return {
        "tokenizer": tokenizer,
        "embedding_layer": emb_layer,
        "sample_ids": sample_ids,
        "decoded_sample": tokenizer.decode(sample_ids),
        "embedding_shape": tuple(embedded.shape),
    }


# ---------------------------------------------------------------------------
# Milestone 3: Causal Self-Attention, Transformer Block & SportsGPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Configuration for the decoder-only SportsGPT transformer."""

    vocab_size: int = 400
    block_size: int = 64
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with explicit lower-triangular causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
            )
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with GELU nonlinearity."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SportsGPT(nn.Module):
    """Decoder-only GPT language model built from scratch for sports commentary."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenAndPositionalEmbedding(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embedding matrix and output projection reduces parameters
        # and accelerates convergence on compact corpora.
        self.lm_head.weight = self.embeddings.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Autoregressively append up to `max_new_tokens` token IDs to `idx`."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                next_logits = logits[:, -1, :]
                next_token = sample_next_token(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy,
                )
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            if was_training:
                self.train()
        return idx


def milestone_3(config: GPTConfig | None = None) -> dict[str, Any]:
    """Milestone 3 entrypoint: construct SportsGPT and verify causal forward pass."""
    cfg = config or GPTConfig()
    model = SportsGPT(cfg)
    dummy_idx = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    dummy_targets = torch.randint(0, cfg.vocab_size, (2, 16), dtype=torch.long)
    logits, loss = model(dummy_idx, dummy_targets)
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "model": model,
        "config": cfg,
        "logits_shape": tuple(logits.shape),
        "loss": float(loss.item()) if loss is not None else None,
        "param_count": param_count,
    }


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
