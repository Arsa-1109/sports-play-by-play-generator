"""Unit and integration tests for Milestones 1 through 6."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from app.cli import main
from app.core import (
    ARTICLE_BOUNDARY,
    BPETokenizer,
    CausalSelfAttention,
    GPTConfig,
    SportsGPT,
    TokenAndPositionalEmbedding,
    evaluate_baseline,
    generate_text,
    get_batch,
    load_checkpoint,
    load_corpus,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
    normalize_text,
    prepare_corpus_text,
    save_checkpoint,
    train_model,
    train_val_split,
)


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
def test_milestone_5(tmp_path: Path):
    ckpt_path = tmp_path / "gen_test.pt"
    train_model(
        checkpoint_path=ckpt_path,
        vocab_size=360,
        block_size=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        steps=40,
        batch_size=12,
        lr=1.5e-3,
        eval_interval=40,
        seed=7,
        verbose=False,
    )
    model, tok, _, _ = load_checkpoint(ckpt_path)

    # Greedy determinism
    g1 = generate_text(model, tok, prompt="Ninety minutes on the clock", max_new_tokens=20, greedy=True)
    g2 = generate_text(model, tok, prompt="Ninety minutes on the clock", max_new_tokens=20, temperature=0.0)
    assert len(g1) == 1 and g1[0] == g2[0]
    assert g1[0].startswith("Ninety minutes on the clock")

    # Empty prompt handling
    empty_out = generate_text(model, tok, prompt="", max_new_tokens=16, top_k=20, top_p=0.9, seed=11)
    assert len(empty_out) == 1 and len(empty_out[0]) > 0

    # Prompt exceeding block_size (> 32 tokens) must crop cleanly without crashing
    long_prompt = " ".join(["The striker chased the diagonal through ball into the box"] * 6)
    assert len(tok.encode(long_prompt)) > model.config.block_size
    long_out = generate_text(model, tok, prompt=long_prompt, max_new_tokens=15, greedy=True)
    assert len(long_out) == 1 and long_out[0].startswith(long_prompt)

    # Multiple snippets
    multi = milestone_5(prompt="The fast bowler", max_new_tokens=20, num_snippets=3, checkpoint_path=ckpt_path)
    assert len(multi) == 3
    assert all(s.startswith("The fast bowler") for s in multi)


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
def test_milestone_6(tmp_path: Path, capsys, monkeypatch):
    ckpt_path = tmp_path / "cli_ckpt.pt"

    # 1. Explicit train command
    main([
        "train",
        "--steps", "40",
        "--vocab-size", "320",
        "--d-model", "64",
        "--n-heads", "4",
        "--n-layers", "2",
        "--block-size", "48",
        "--checkpoint", str(ckpt_path),
    ])
    train_out = capsys.readouterr().out
    assert "Training complete" in train_out
    assert ckpt_path.exists()

    # 2. Generate command with multiple snippets
    main([
        "generate",
        "--prompt", "With four seconds remaining",
        "--max-tokens", "24",
        "--num-snippets", "2",
        "--checkpoint", str(ckpt_path),
        "--seed", "99",
    ])
    gen_out = capsys.readouterr().out
    assert "[Snippet 1] With four seconds remaining" in gen_out
    assert "[Snippet 2] With four seconds remaining" in gen_out

    # 3. Auto-train fallback when generate is run against a missing checkpoint
    auto_ckpt = tmp_path / "auto_ckpt.pt"
    main([
        "generate",
        "--prompt", "Bottom of the ninth",
        "--max-tokens", "16",
        "--greedy",
        "--checkpoint", str(auto_ckpt),
    ])
    auto_out = capsys.readouterr().out
    assert auto_ckpt.exists()
    assert "[Snippet 1] Bottom of the ninth" in auto_out

    # 4. Interactive REPL command (tip, placeholder stripping, standalone + inline commands)
    inputs = iter([
        "/temp 0.6",
        "/greedy <The point guard crossed half-court>",
        "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    main([
        "interactive",
        "--checkpoint", str(ckpt_path),
        "--max-tokens", "15",
    ])
    inter_out = capsys.readouterr().out
    assert "Tip:" in inter_out
    assert "Temperature set to 0.6." in inter_out
    assert "[Snippet 1] The point guard crossed half-court" in inter_out

    # 5. Baseline evaluation CLI & programmatic check
    report = evaluate_baseline(checkpoint_path=ckpt_path, train_if_missing=False)
    assert report["val_eval_tokens"] > 100
    assert report["beats_unigram"] is True
    main(["eval", "--checkpoint", str(ckpt_path), "--json"])
    eval_json = json.loads(capsys.readouterr().out)
    assert [m["name"] for m in eval_json["models"]] == [
        "Uniform Random",
        "Unigram (Laplace)",
        "Bigram (Laplace)",
        "SportsGPT (Ours)",
    ]

    # 6. Clean error handling on invalid flags / missing corpus (exit code 2, no raw traceback)
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "--max-tokens", "0"])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as missing_exc:
        main(["train", "--corpus", str(tmp_path / "does_not_exist.txt"), "--steps", "5"])
    assert missing_exc.value.code == 2

    # 7. Programmatic milestone_6 helper
    res = milestone_6(prompt="Serving for the championship", max_new_tokens=15, checkpoint_path=ckpt_path)
    assert len(res) == 1 and res[0].startswith("Serving for the championship")
