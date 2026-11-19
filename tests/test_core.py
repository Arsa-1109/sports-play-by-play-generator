"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from pathlib import Path
from app.core import ARTICLE_BOUNDARY, load_corpus, milestone_1, normalize_text, prepare_corpus_text, train_val_split
import torch
from app.core import BPETokenizer, TokenAndPositionalEmbedding, milestone_2

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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Train the GPT model using the sports text data.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Develop a text generation function to produce short commentary snippets.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Create a CLI where users can input a prompt and receive generated commentary.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

