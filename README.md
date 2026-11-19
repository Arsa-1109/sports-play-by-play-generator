# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/pzuatu3vjh/badge.svg)](https://ask.thili.ai/b/pzuatu3vjh)

A self-contained command-line sports commentary generator powered by a decoder-only GPT transformer (`SportsGPT`) and byte-level Byte-Pair Encoding (`BPETokenizer`) built from scratch in PyTorch. Trained on a multi-sport corpus of news reports and play-by-play commentary, it supports training, baseline evaluation, single-shot generation, and interactive play-by-play prompting directly on your laptop.

## Key Features

- **Lossless Byte-Level BPE Tokenizer (`BPETokenizer`):** Starts from UTF-8 byte base tokens (`0..255`) and learns frequent adjacent pair merges, guaranteeing zero `<UNK>` failures on unseen words, punctuation, scores, or Unicode characters.
- **From-Scratch Decoder-Only Transformer (`SportsGPT`):** Combines learned token and positional embeddings (with weight tying to `lm_head`), multi-head `CausalSelfAttention` using explicit lower-triangular `-inf` masking (`torch.tril`), Pre-LayerNorm `TransformerBlock`s, and GELU `FeedForward` layers.
- **AdamW Training with Warmup & Cosine Decay:** Includes next-token batching, gradient norm clipping, periodic train/validation loss tracking, single-threaded CPU execution tuning, and `.pt` checkpoint persistence bundling weights, config, and BPE merges.
- **Flexible Decoding & Forgiving CLI:** Supports deterministic greedy decoding (`--greedy` / `--temperature 0`), temperature scaling, top-k filtering, nucleus (`--top-p`) sampling, multi-snippet generation, and an interactive REPL with inline commands and placeholder stripping.
- **Quantitative Benchmarking (`eval`):** Compares `SportsGPT` deterministically against Uniform, Laplace-smoothed Unigram, and Laplace-smoothed Bigram baselines on the held-out validation split.

## Project Structure

```text
sports-play-by-play-generator/
├── .github/workflows/ci.yml   # GitHub Actions pytest check
├── app/
│   ├── __init__.py
│   ├── cli.py                 # CLI commands: hello, train, generate, eval, interactive
│   ├── core.py                # Corpus loader, BPETokenizer, SportsGPT, trainer, eval, sampling
│   ├── corpus.py              # Built-in 34-article multi-sport commentary corpus
│   ├── llm.py                 # Optional OpenAI-compatible .env configuration helper
│   └── storage.py             # Lightweight JSON key-value store helper
├── tests/
│   ├── test_core.py           # Unit & integration tests for BPE, attention, training, decoding, CLI & eval
│   └── test_smoke.py          # Plumbing smoke tests (Store, settings, hello CLI)
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Quickstart

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app.cli hello
python -m app.cli train --steps 350 --checkpoint data/sports_gpt.pt
python -m app.cli eval --checkpoint data/sports_gpt.pt
python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --seed 42
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app.cli hello
python -m app.cli train --steps 350 --checkpoint data/sports_gpt.pt
python -m app.cli eval --checkpoint data/sports_gpt.pt
python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --seed 42
```

> Note: Running `generate`, `eval`, or `interactive` without an existing checkpoint automatically trains a baseline model and saves it to `data/sports_gpt.pt`.

## CLI Reference

| Command | Description | Key Options |
| :--- | :--- | :--- |
| `python -m app.cli hello` | Verify CLI and package wiring | — |
| `python -m app.cli train` | Train `BPETokenizer` + `SportsGPT` and save checkpoint | `--steps`, `--vocab-size`, `--d-model`, `--n-heads`, `--n-layers`, `--block-size`, `--batch-size`, `--lr`, `--corpus`, `--checkpoint`, `--seed` |
| `python -m app.cli generate` | Generate commentary snippets from a prompt | `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--greedy`, `--num-snippets`, `--checkpoint`, `--seed` |
| `python -m app.cli eval` | Benchmark `SportsGPT` vs. Uniform, Unigram, and Bigram baselines | `--checkpoint`, `--corpus`, `--steps`, `--json` |
| `python -m app.cli interactive` | Launch interactive prompt REPL (supports `/greedy`, `/sample`, `/temp`, `/snippets`, `/max-tokens`, `/eval`, `/help`, `quit`) | Same generation options as `generate` |

## Sample Output

```text
$ python -m app.cli hello
Sports Play-by-Play Generator is ready. Open README.md for your milestones.

$ python -m app.cli train --steps 350 --checkpoint data/sports_gpt.pt
step    0 | train_loss=5.9968 | val_loss=6.0066
step  175 | train_loss=3.2621 | val_loss=3.6393
step  350 | train_loss=2.8597 | val_loss=3.4385
Training complete | tokens=12168 | initial_loss=5.9968 -> final_train_loss=2.8597 | final_val_loss=3.4385 | saved=data\sports_gpt.pt

$ python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --checkpoint data/sports_gpt.pt --seed 42
[Snippet 1] Ninety minutes on the clock, and the strikerveter corners, the clock, the fuaies into the four, a crips and set.
[Snippet 2] Ninety minutes on the clock, and the strikers cutes for the slightservy up onearing ender sward-backs a foubort the juast the spox.

$ python -m app.cli interactive --checkpoint data/sports_gpt.pt --max-tokens 35
Interactive Sports Play-by-Play Generator
Tip: Type a game scenario prompt (e.g. "Ninety minutes on the clock,") or command (/greedy, /sample, /temp 0.7, /snippets 2, /max-tokens 60, /eval, /help, quit).
Prompt> /temp 0.6
Temperature set to 0.6.
Prompt> /greedy <With four seconds remaining on the game clock>
[Snippet 1] With four seconds remaining on the game clocked the six-pped the sield.
Prompt> quit
```

## Baseline Evaluation & Limitations

Held-out validation benchmark (`python -m app.cli eval --checkpoint data/sports_gpt.pt`, evaluated deterministically across all `1,215` next-token targets in the 10% validation split with `vocab_size=400`, after 350 training steps):

```text
Evaluation on held-out validation split (1215 tokens, vocab=400):
Model                 Cross-Entropy   Perplexity   Top-1 Acc   Top-5 Acc
------------------------------------------------------------------------
Uniform Random               5.9915       400.00       0.25%       1.25%
Unigram (Laplace)            4.9983       148.16       3.13%      13.25%
Bigram (Laplace)             4.3226        75.39      15.47%      44.86%
SportsGPT (Ours)             3.4425        31.27      18.52%      45.19%
------------------------------------------------------------------------
SportsGPT PPL reduction: +78.90% vs Unigram | +58.53% vs Bigram
```

### Limitations

1. **Compact Training Corpus & Subword Artifacts:** The built-in 34-article corpus (`12,168` BPE tokens) is intentionally small so it trains from scratch on a laptop CPU in ~1 minute. At this scale, subword BPE pieces occasionally recombine into non-dictionary words (for example `strikerveter` or `slightservy`) during sampling.
2. **Short Context Window (`block_size=64`):** Inputs exceeding 64 BPE tokens are cropped to the most recent 64 tokens, limiting long-range continuity across multi-paragraph game narratives.
3. **No Factual Scoreboard or Rules Grounding:** `SportsGPT` models token co-occurrence statistics only; it does not track legal game state, player rosters, or score math and will invent fictional match details.
4. **CPU Training Speed:** Training time scales with `--steps`, `--batch-size`, and `--d-model`. Large hyperparameter sweeps on CPU will take several minutes without GPU acceleration.

## Testing

All tests run offline without `.env` or external API keys:

```bash
pytest -v
```

Observed suite result: `9 passed, 0 skipped, 0 failed` (`test_milestone_1` through `test_milestone_6` in `tests/test_core.py`, plus `test_the_store_keeps_what_you_put_in_it`, `test_model_settings_default_to_a_local_ollama`, and `test_the_command_line_runs` in `tests/test_smoke.py`).

## Configuration

Optional LLM settings for `app/llm.py` live in `.env` (copied from `.env.example`), which is ignored by `.gitignore`. Never commit `.env` or API keys. Core training, evaluation, and play-by-play generation in `app/core.py` run completely offline using PyTorch.
