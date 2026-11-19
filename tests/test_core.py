"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from pathlib import Path
from app.core import ARTICLE_BOUNDARY, load_corpus, milestone_1, normalize_text, prepare_corpus_text, train_val_split
import torch
from app.core import BPETokenizer, TokenAndPositionalEmbedding, milestone_2
import math
from app.core import CausalSelfAttention, GPTConfig, milestone_3
from app.core import get_batch, load_checkpoint, milestone_4, train_model

# Milestone 1: Gather a small corpus of sports commentary or news articles.
def test_milestone_1(tmp_path: Path):
    info = milestone_1()
    assert info["num_articles"] >= 30
    assert 20_000 <= info["total_chars"] <= 60_000
    assert ARTICLE_BOUNDARY in info["text"]

    # Whitespace normalization
    messy = "  Goal!\r\n\r\n\r\n   Top   corner \t finish.  "
    assert normalize_text(messy) == "Goal!\n\nTop corner finish."

    # Custom corpus file loading
    custom_file = tmp_path / "custom_sports.txt"
    custom_file.write_text("First half opener.\n---\nSecond half winner.", encoding="utf-8")
    loaded = load_corpus(custom_file)
    assert loaded == ["First half opener.", "Second half winner."]
    joined = prepare_corpus_text(loaded)
    assert ARTICLE_BOUNDARY in joined

    # Deterministic train/val split
    seq = list(range(100))
    tr1, va1 = train_val_split(seq, val_ratio=0.1)
    tr2, va2 = train_val_split(seq, val_ratio=0.1)
    assert torch.equal(tr1, tr2) and torch.equal(va1, va2)
    assert len(tr1) == 90 and len(va1) == 10


# Milestone 2: Implement the byte-pair tokenizer and create embeddings from the sports text.
def test_milestone_2():
    corpus = milestone_1()["text"]
    tok = BPETokenizer().train(corpus, vocab_size=400)
    assert tok.vocab_size == 400
    assert len(tok.merges) == 400 - 256

    # Edge cases: empty, single char, punctuation, unseen Unicode/emoji
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    for sample in [
        "a",
        "Goal! 90+3' — top-corner volley!",
        "⚽🏀🏏⚾🎾 café naïve résumé — 100%",
    ]:
        ids = tok.encode(sample)
        assert len(ids) >= 1
        assert tok.decode(ids) == sample

    # Serialization roundtrip
    restored = BPETokenizer.from_dict(tok.to_dict())
    assert restored.vocab_size == tok.vocab_size
    assert restored.encode("striker volley goal") == tok.encode("striker volley goal")

    # Token + positional embeddings
    emb = TokenAndPositionalEmbedding(vocab_size=tok.vocab_size, block_size=32, d_model=64, dropout=0.0)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = emb(x)
    assert out.shape == (1, 4, 64)
    with pytest.raises(ValueError, match="exceeds block_size"):
        emb(torch.zeros((1, 33), dtype=torch.long))

    m2 = milestone_2(corpus, vocab_size=320, d_model=64, block_size=48)
    assert m2["embedding_shape"][0] == 1 and m2["embedding_shape"][2] == 64


# Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
def test_milestone_3():
    cfg = GPTConfig(vocab_size=300, block_size=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    attn.eval()

    # Verify strict causality: perturbing future token at index 5 must not alter output at indices 0..4
    x1 = torch.randn(1, 8, cfg.d_model)
    x2 = x1.clone()
    x2[:, 5:, :] += 10.0
    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)
    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6)
    assert not torch.allclose(y1[:, 5:, :], y2[:, 5:, :], atol=1e-4)

    # Full SportsGPT causal check + forward shapes
    m3 = milestone_3(cfg)
    assert m3["logits_shape"] == (2, 16, 300)
    assert m3["loss"] is not None and math.isfinite(m3["loss"])
    assert m3["param_count"] > 0


# Milestone 4: Train the GPT model using the sports text data.
def test_milestone_4(tmp_path: Path):
    # Check get_batch target alignment (next-token shift)
    seq = torch.arange(50, dtype=torch.long)
    xb, yb = get_batch(seq, block_size=8, batch_size=4)
    assert xb.shape == (4, 8) and yb.shape == (4, 8)
    assert torch.equal(xb[:, 1:], yb[:, :-1])

    ckpt_path = tmp_path / "model.pt"
    result = train_model(
        checkpoint_path=ckpt_path,
        vocab_size=360,
        block_size=48,
        d_model=64,
        n_heads=4,
        n_layers=2,
        steps=50,
        batch_size=12,
        lr=1.5e-3,
        eval_interval=50,
        seed=42,
        verbose=False,
    )
    assert ckpt_path.exists()
    assert result["final_train_loss"] < result["initial_loss"]

    # Checkpoint load roundtrip
    model, tok, cfg, meta = load_checkpoint(ckpt_path)
    assert cfg.vocab_size == tok.vocab_size
    assert meta["steps"] == 50
    probe = torch.tensor([[10, 20, 30]], dtype=torch.long)
    result["model"].eval()
    with torch.no_grad():
        logits_orig, _ = result["model"](probe)
        logits_loaded, _ = model(probe)
    assert torch.allclose(logits_orig, logits_loaded, atol=1e-6)

    m4 = milestone_4(steps=30, checkpoint_path=tmp_path / "m4.pt")
    assert m4["final_train_loss"] < m4["initial_loss"]


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

