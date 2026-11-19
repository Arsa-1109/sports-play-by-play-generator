"""Core implementation of Sports Play-by-Play Generator (GPT from scratch)."""
from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)

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

    def train(self, text: str, vocab_size: int = 900) -> "BPETokenizer":
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

        # Maintain incremental pair frequencies and an inverted index so each merge
        # updates only the chunks containing the winning pair.
        pair_freqs: Counter[tuple[int, int]] = Counter()
        pair_to_chunks: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for seq, freq in chunk_counts.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_freqs[pair] += freq
                pair_to_chunks.setdefault(pair, set()).add(seq)

        num_merges = vocab_size - 256
        for step in range(num_merges):
            if not pair_freqs:
                break

            # Deterministic tie-breaking: highest frequency, then lexicographically smallest pair.
            best_pair = min(pair_freqs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = 256 + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            affected_seqs = list(pair_to_chunks.pop(best_pair, ()))
            for seq in affected_seqs:
                freq = chunk_counts.pop(seq, 0)
                if freq <= 0:
                    continue

                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_freqs[pair] -= freq
                    if pair_freqs[pair] <= 0:
                        pair_freqs.pop(pair, None)
                    chunk_set = pair_to_chunks.get(pair)
                    if chunk_set is not None:
                        chunk_set.discard(seq)
                        if not chunk_set:
                            pair_to_chunks.pop(pair, None)

                merged = tuple(_merge_pair_in_sequence(list(seq), best_pair, new_id))
                chunk_counts[merged] += freq
                for i in range(len(merged) - 1):
                    pair = (merged[i], merged[i + 1])
                    pair_freqs[pair] += freq
                    pair_to_chunks.setdefault(pair, set()).add(merged)

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
    vocab_size: int = 900,
    d_model: int = 128,
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

    vocab_size: int = 900
    block_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = False


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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        if config.tie_weights:
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
        temperature: float = 0.65,
        top_k: int | None = 25,
        top_p: float | None = 0.88,
        greedy: bool = False,
        repetition_penalty: float = 1.15,
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
                    repetition_penalty=repetition_penalty,
                    recent_tokens=idx[:, -32:],
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


# ---------------------------------------------------------------------------
# Milestone 4: Training Loop, Loss Calculation & Checkpointing
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, next-token target) windows from 1D token tensor."""
    if data.numel() < 2:
        raise ValueError("Data tensor must have at least 2 tokens.")
    eff_block = min(block_size, data.numel() - 1)
    max_start = data.numel() - eff_block
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + eff_block] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + eff_block] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: SportsGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Estimate average train and validation cross-entropy loss without gradients."""
    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}
    try:
        for split_name, split_data in (("train", train_data), ("val", val_data)):
            batch_losses = []
            for _ in range(eval_iters):
                xb, yb = get_batch(split_data, block_size, batch_size, device=device)
                _, loss = model(xb, yb)
                assert loss is not None
                batch_losses.append(float(loss.item()))
            losses[split_name] = sum(batch_losses) / len(batch_losses)
    finally:
        if was_training:
            model.train()
    return losses


def save_checkpoint(
    path: str | Path,
    model: SportsGPT,
    tokenizer: BPETokenizer,
    config: GPTConfig,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights, GPTConfig, and BPE tokenizer state to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SportsGPT, BPETokenizer, GPTConfig, dict[str, Any]]:
    """Restore SportsGPT, BPETokenizer, GPTConfig, and metadata from checkpoint."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    tokenizer = BPETokenizer.from_dict(payload["tokenizer"])
    model = SportsGPT(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, config, payload.get("metadata", {})


@contextmanager
def _cpu_single_thread(device: str):
    """Avoid OpenMP thread-sync overhead on small CPU transformer matrices."""
    if str(device).startswith("cpu"):
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(prev)
    else:
        yield


def _cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup over first 10% of steps followed by cosine decay to 10% of base_lr."""
    warmup_steps = max(1, total_steps // 10)
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (0.1 + 0.9 * cosine)


def train_model(
    corpus_path: str | Path | None = None,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    vocab_size: int = 900,
    block_size: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    steps: int = 400,
    batch_size: int = 16,
    lr: float = 3e-3,
    eval_interval: int = 100,
    grad_clip: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end training pipeline: corpus -> BPE -> SportsGPT -> checkpoint."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    torch.manual_seed(seed)

    articles = load_corpus(corpus_path)
    corpus_text = prepare_corpus_text(articles)
    tokenizer = BPETokenizer().train(corpus_text, vocab_size=vocab_size)
    token_ids = tokenizer.encode(corpus_text)
    train_data, val_data = train_val_split(token_ids, val_ratio=0.1)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    model = SportsGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    with _cpu_single_thread(device):
        initial_losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=3, device=device)
        initial_loss = initial_losses["train"]
        history: list[dict[str, float | int]] = [
            {"step": 0, "train_loss": initial_losses["train"], "val_loss": initial_losses["val"]}
        ]
        if verbose:
            print(
                f"step    0 | train_loss={initial_losses['train']:.4f} | val_loss={initial_losses['val']:.4f}"
            )

        model.train()
        for step in range(1, steps + 1):
            step_lr = _cosine_lr(step, steps, lr)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            xb, yb = get_batch(train_data, block_size, batch_size, device=device)
            _, loss = model(xb, yb)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % eval_interval == 0 or step == steps:
                losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=3, device=device)
                history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})
                if verbose:
                    print(
                        f"step {step:4d} | train_loss={losses['train']:.4f} | val_loss={losses['val']:.4f}"
                    )

    final_train_loss = float(history[-1]["train_loss"])
    final_val_loss = float(history[-1]["val_loss"])
    metadata = {
        "steps": steps,
        "initial_loss": initial_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "num_tokens": len(token_ids),
        "num_articles": len(articles),
    }

    saved_path: str | None = None
    if checkpoint_path is not None:
        saved_path = str(save_checkpoint(checkpoint_path, model, tokenizer, config, metadata=metadata))

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "initial_loss": initial_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "history": history,
        "checkpoint_path": saved_path,
        "num_tokens": len(token_ids),
    }


def milestone_4(steps: int = 80, checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH) -> dict[str, Any]:
    """Milestone 4 entrypoint: train SportsGPT on the sports corpus and track loss."""
    return train_model(steps=steps, checkpoint_path=checkpoint_path, verbose=False)


# ---------------------------------------------------------------------------
# Milestone 5: Text Generation (Greedy Decoding, Temperature, Top-k, Top-p)
# ---------------------------------------------------------------------------

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.65,
    top_k: int | None = 25,
    top_p: float | None = 0.88,
    greedy: bool = False,
    repetition_penalty: float = 1.15,
    recent_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the next token ID from `(B, vocab_size)` logits via greedy or filtered sampling."""
    if repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

    adjusted = logits
    if repetition_penalty > 1.0 and recent_tokens is not None and recent_tokens.numel() > 0:
        adjusted = logits.clone()
        for row_idx in range(adjusted.size(0)):
            seen_ids = torch.unique(recent_tokens[row_idx])
            row_vals = adjusted[row_idx, seen_ids]
            adjusted[row_idx, seen_ids] = torch.where(
                row_vals < 0,
                row_vals * repetition_penalty,
                row_vals / repetition_penalty,
            )

    if greedy or temperature <= 0.0:
        return torch.argmax(adjusted, dim=-1, keepdim=True)

    scaled = adjusted / temperature
    vocab_size = scaled.size(-1)

    if top_k is not None and 0 < top_k < vocab_size:
        top_vals, _ = torch.topk(scaled, top_k, dim=-1)
        cutoff = top_vals[:, [-1]]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens once cumulative probability exceeds top_p, always retaining top-1.
        sorted_remove = (cumulative_probs - sorted_probs) > top_p
        sorted_remove[:, 0] = False
        remove_mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(1, sorted_indices, sorted_remove)
        scaled = scaled.masked_fill(remove_mask, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _trim_to_sentence_end(text: str) -> str:
    """Trim generated continuation to the last complete sentence boundary when available."""
    if not text:
        return text
    min_cut = int(len(text) * 0.35)
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] in ".!?" and (idx + 1 == len(text) or text[idx + 1].isspace()):
            if idx >= min_cut:
                return text[: idx + 1]
            break
    cleaned = re.sub(r"[-\s]+$", "", text)
    return cleaned


def generate_text(
    model: SportsGPT,
    tokenizer: BPETokenizer,
    prompt: str = "",
    max_new_tokens: int = 80,
    temperature: float = 0.65,
    top_k: int | None = 25,
    top_p: float | None = 0.88,
    greedy: bool = False,
    repetition_penalty: float = 1.15,
    num_snippets: int = 1,
    stop_on_article_boundary: bool = True,
    seed: int | None = None,
    device: str = "cpu",
) -> list[str]:
    """Generate one or more sports commentary snippets from a prompt."""
    if num_snippets < 1:
        raise ValueError(f"num_snippets must be >= 1, got {num_snippets}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

    if seed is not None:
        torch.manual_seed(seed)

    prompt_ids = tokenizer.encode(prompt)
    # Handle empty prompt gracefully by seeding with a newline byte token.
    seed_ids = prompt_ids if prompt_ids else [ord("\n")]
    prompt_len = len(prompt_ids)

    model.eval()
    snippets: list[str] = []
    for _ in range(num_snippets):
        idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
        out_ids = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            repetition_penalty=repetition_penalty,
        )[0].tolist()

        if prompt_ids:
            generated_ids = out_ids[prompt_len:]
            continuation = tokenizer.decode(generated_ids)
            if stop_on_article_boundary and ARTICLE_BOUNDARY in continuation:
                head = continuation.split(ARTICLE_BOUNDARY, 1)[0]
                if head.strip():
                    continuation = head
            continuation = _trim_to_sentence_end(continuation)
            snippet = f"{prompt}{continuation}".strip()
        else:
            decoded = tokenizer.decode(out_ids[len(seed_ids) :])
            if stop_on_article_boundary and ARTICLE_BOUNDARY in decoded:
                parts = [p.strip() for p in decoded.split(ARTICLE_BOUNDARY) if p.strip()]
                decoded = parts[0] if parts else decoded
            snippet = _trim_to_sentence_end(decoded).strip()

        snippets.append(snippet)
    return snippets


def ensure_trained_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    steps: int = 300,
    verbose: bool = True,
) -> Path:
    """Return existing checkpoint path or fast-train a baseline model if missing."""
    path = Path(checkpoint_path)
    if path.exists():
        return path
    if verbose:
        print(f"Checkpoint '{path}' not found. Auto-training baseline SportsGPT ({steps} steps)...")
    train_model(steps=steps, checkpoint_path=path, verbose=verbose)
    return path


@torch.no_grad()
def evaluate_baseline(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    corpus_path: str | Path | None = None,
    val_ratio: float = 0.1,
    steps: int = 400,
    train_if_missing: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Benchmark SportsGPT against Uniform, Unigram, and Bigram baselines on held-out tokens."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        train_model(corpus_path=corpus_path, checkpoint_path=ckpt, steps=steps, verbose=False, device=device)

    model, tokenizer, config, metadata = load_checkpoint(ckpt, device=device)
    articles = load_corpus(corpus_path)
    corpus_text = prepare_corpus_text(articles)
    token_ids = tokenizer.encode(corpus_text)
    train_data, val_data = train_val_split(token_ids, val_ratio=val_ratio)

    train_list = train_data.tolist()
    val_list = val_data.tolist()
    n_eval = len(val_list) - 1
    if n_eval < 1:
        raise ValueError("Validation split must contain at least 2 tokens for evaluation.")

    vocab_size = tokenizer.vocab_size

    # 1. Uniform random baseline
    uniform_ce = math.log(vocab_size)
    uniform_ppl = float(vocab_size)
    uniform_top1 = 1.0 / vocab_size
    uniform_top5 = min(5, vocab_size) / vocab_size

    # 2. Laplace-smoothed Unigram frequency baseline (fit strictly on train_data)
    uni_counts = Counter(train_list)
    uni_denom = len(train_list) + vocab_size
    uni_top5_tokens = [tok for tok, _ in uni_counts.most_common(5)]
    uni_top1_tok = uni_top5_tokens[0]
    uni_top5_set = set(uni_top5_tokens)

    uni_nll = 0.0
    uni_t1 = 0
    uni_t5 = 0
    for target in val_list[1:]:
        p = (uni_counts[target] + 1.0) / uni_denom
        uni_nll -= math.log(p)
        if target == uni_top1_tok:
            uni_t1 += 1
        if target in uni_top5_set:
            uni_t5 += 1
    unigram_ce = uni_nll / n_eval
    unigram_ppl = math.exp(unigram_ce)

    # 3. Laplace-smoothed Bigram Markov baseline (fit strictly on train_data)
    alpha = 0.5
    ctx_counts = Counter(train_list[:-1])
    bi_counts: dict[int, Counter[int]] = {}
    for a, b in zip(train_list[:-1], train_list[1:]):
        bi_counts.setdefault(a, Counter())[b] += 1

    bi_nll = 0.0
    bi_t1 = 0
    bi_t5 = 0
    top5_cache: dict[int, list[int]] = {}
    for prev_tok, target in zip(val_list[:-1], val_list[1:]):
        next_counter = bi_counts.get(prev_tok)
        denom = ctx_counts[prev_tok] + alpha * vocab_size
        c = next_counter[target] if next_counter is not None else 0
        bi_nll -= math.log((c + alpha) / denom)
        if prev_tok not in top5_cache:
            if next_counter:
                top5_cache[prev_tok] = [tok for tok, _ in next_counter.most_common(5)]
            else:
                top5_cache[prev_tok] = uni_top5_tokens
        ranked = top5_cache[prev_tok]
        if target == ranked[0]:
            bi_t1 += 1
        if target in ranked:
            bi_t5 += 1
    bigram_ce = bi_nll / n_eval
    bigram_ppl = math.exp(bigram_ce)

    # 4. SportsGPT deterministic evaluation across every token in val_data[1:]
    model.eval()
    gpt_nll = 0.0
    gpt_t1 = 0
    gpt_t5 = 0
    with _cpu_single_thread(device):
        pos = 1
        while pos < len(val_list):
            end = min(len(val_list), pos + config.block_size)
            chunk = val_data[pos - 1 : end].to(device)
            inputs = chunk[:-1].unsqueeze(0)
            targets = chunk[1:]
            logits, _ = model(inputs)
            step_logits = logits[0]
            log_probs = F.log_softmax(step_logits, dim=-1)
            gpt_nll -= float(log_probs[torch.arange(targets.size(0), device=device), targets].sum().item())
            top5_preds = torch.topk(step_logits, k=min(5, vocab_size), dim=-1).indices
            gpt_t1 += int((top5_preds[:, 0] == targets).sum().item())
            gpt_t5 += int((top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item())
            pos = end

    gpt_ce = gpt_nll / n_eval
    gpt_ppl = math.exp(gpt_ce)

    models_summary = [
        {
            "name": "Uniform Random",
            "cross_entropy": round(uniform_ce, 4),
            "perplexity": round(uniform_ppl, 2),
            "top1_acc": round(uniform_top1, 4),
            "top5_acc": round(uniform_top5, 4),
        },
        {
            "name": "Unigram (Laplace)",
            "cross_entropy": round(unigram_ce, 4),
            "perplexity": round(unigram_ppl, 2),
            "top1_acc": round(uni_t1 / n_eval, 4),
            "top5_acc": round(uni_t5 / n_eval, 4),
        },
        {
            "name": "Bigram (Laplace)",
            "cross_entropy": round(bigram_ce, 4),
            "perplexity": round(bigram_ppl, 2),
            "top1_acc": round(bi_t1 / n_eval, 4),
            "top5_acc": round(bi_t5 / n_eval, 4),
        },
        {
            "name": "SportsGPT (Ours)",
            "cross_entropy": round(gpt_ce, 4),
            "perplexity": round(gpt_ppl, 2),
            "top1_acc": round(gpt_t1 / n_eval, 4),
            "top5_acc": round(gpt_t5 / n_eval, 4),
        },
    ]

    return {
        "checkpoint_path": str(ckpt),
        "vocab_size": vocab_size,
        "train_tokens": len(train_list),
        "val_eval_tokens": n_eval,
        "trained_steps": metadata.get("steps"),
        "models": models_summary,
        "ppl_reduction_vs_unigram_pct": round((unigram_ppl - gpt_ppl) / unigram_ppl * 100.0, 2),
        "ppl_reduction_vs_bigram_pct": round((bigram_ppl - gpt_ppl) / bigram_ppl * 100.0, 2),
        "beats_unigram": gpt_ce < unigram_ce,
        "beats_bigram": gpt_ce < bigram_ce,
    }


def milestone_5(
    prompt: str = "Ninety minutes on the clock, and the striker",
    max_new_tokens: int = 60,
    num_snippets: int = 2,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> list[str]:
    """Milestone 5 entrypoint: generate sports commentary snippets from a prompt."""
    ckpt = ensure_trained_checkpoint(checkpoint_path, verbose=False)
    model, tokenizer, _, _ = load_checkpoint(ckpt)
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_snippets=num_snippets,
    )


# ---------------------------------------------------------------------------
# Milestone 6: CLI Interactive & Programmatic Entrypoint
# ---------------------------------------------------------------------------

def milestone_6(
    prompt: str = "Bottom of the ninth inning, two outs,",
    max_new_tokens: int = 70,
    temperature: float = 0.65,
    top_k: int | None = 25,
    top_p: float | None = 0.88,
    greedy: bool = False,
    repetition_penalty: float = 1.15,
    num_snippets: int = 1,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> list[str]:
    """Milestone 6 entrypoint: generate commentary ready for CLI display."""
    ckpt = ensure_trained_checkpoint(checkpoint_path, verbose=False)
    model, tokenizer, _, _ = load_checkpoint(ckpt)
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        num_snippets=num_snippets,
    )
