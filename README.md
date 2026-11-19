# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/pzuatu3vjh/badge.svg)](https://ask.thili.ai/b/pzuatu3vjh)

A self-contained command-line sports commentary generator powered by a decoder-only GPT transformer (`SportsGPT`) and byte-level Byte-Pair Encoding (`BPETokenizer`) built from scratch in PyTorch. Trained on a multi-sport corpus of news reports and play-by-play commentary across football, basketball, cricket, baseball, and tennis, it supports training, baseline evaluation, single-shot generation, and interactive play-by-play prompting directly on your laptop.

## Key Features

- **Lossless Byte-Level BPE Tokenizer (`BPETokenizer`):** Starts from UTF-8 byte base tokens (`0..255`) and uses an incremental inverted-index pair counter to learn merges up to `vocab_size=900` in under a second, guaranteeing zero `<UNK>` failures on unseen words, punctuation, scores, or Unicode characters.
- **From-Scratch Decoder-Only Transformer (`SportsGPT`):** Combines learned token and positional embeddings (`d_model=128`, `block_size=64`), 2 Pre-LayerNorm `TransformerBlock`s with 4-head `CausalSelfAttention` using explicit lower-triangular `-inf` masking (`torch.tril`), GELU `FeedForward` layers, and an untied `lm_head` projection with bias for fast convergence.
- **AdamW Training with Warmup & Cosine Decay:** Includes next-token batching, gradient norm clipping, periodic train/validation loss tracking, single-threaded CPU execution tuning, and `.pt` checkpoint persistence bundling weights, config, and BPE merges.
- **Fluent Decoding & Forgiving CLI:** Supports deterministic greedy decoding (`--greedy` / `--temperature 0`), temperature scaling, top-k filtering, nucleus (`--top-p`) sampling, CTRL-style `--repetition-penalty`, sentence-boundary trimming, multi-snippet generation, and an interactive REPL with inline commands and placeholder stripping.
- **Quantitative Benchmarking (`eval`):** Compares `SportsGPT` deterministically against Uniform, Laplace-smoothed Unigram, and Laplace-smoothed Bigram baselines on the held-out validation split.

## Project Structure

```text
sports-play-by-play-generator/
├── .github/workflows/ci.yml   # GitHub Actions pytest check
├── app/
│   ├── __init__.py
│   ├── cli.py                 # CLI commands: hello, train, generate, eval, interactive
│   ├── core.py                # Corpus loader, BPETokenizer, SportsGPT, trainer, eval, sampling
│   ├── corpus.py              # Built-in 186-article multi-sport commentary corpus
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
python -m app.cli train --steps 400 --checkpoint data/sports_gpt.pt
python -m app.cli eval --checkpoint data/sports_gpt.pt
python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --seed 42
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app.cli hello
python -m app.cli train --steps 400 --checkpoint data/sports_gpt.pt
python -m app.cli eval --checkpoint data/sports_gpt.pt
python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --seed 42
```

> Note: Running `generate`, `eval`, or `interactive` without an existing checkpoint automatically trains a baseline model and saves it to `data/sports_gpt.pt`.

## CLI Reference

| Command | Description | Key Options |
| :--- | :--- | :--- |
| `python -m app.cli hello` | Verify CLI and package wiring | — |
| `python -m app.cli train` | Train `BPETokenizer` + `SportsGPT` and save checkpoint | `--steps`, `--vocab-size`, `--d-model`, `--n-heads`, `--n-layers`, `--block-size`, `--batch-size`, `--lr`, `--corpus`, `--checkpoint`, `--seed` |
| `python -m app.cli generate` | Generate commentary snippets from a prompt | `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--greedy`, `--num-snippets`, `--checkpoint`, `--seed` |
| `python -m app.cli eval` | Benchmark `SportsGPT` vs. Uniform, Unigram, and Bigram baselines | `--checkpoint`, `--corpus`, `--steps`, `--json` |
| `python -m app.cli interactive` | Launch interactive prompt REPL (supports `/greedy`, `/sample`, `/temp`, `/snippets`, `/max-tokens`, `/eval`, `/help`, `quit`) | Same generation options as `generate` |

## Sample Output

```text
$ python -m app.cli hello
Sports Play-by-Play Generator is ready. Open README.md for your milestones.

$ python -m app.cli train --steps 400 --checkpoint data/sports_gpt.pt
step    0 | train_loss=6.8205 | val_loss=6.8249
step  200 | train_loss=1.0767 | val_loss=0.8974
step  400 | train_loss=0.6327 | val_loss=0.3324
Training complete | tokens=28949 | initial_loss=6.8205 -> final_train_loss=0.6327 | final_val_loss=0.3324 | saved=data\sports_gpt.pt

$ python -m app.cli generate --prompt "Ninety minutes on the clock, and the striker" --max-tokens 60 --num-snippets 2 --seed 42
[Snippet 1] Ninety minutes on the clock, and the striker ghosted past the back four to meet a whipped cross from the right flank. He curled a low shot around the rushing goalkeeper into the bottom corner as the crowd erupted.
[Snippet 2] Ninety minutes on the clock, and the striker ghosted past the back four to meet a whipped cross from twelve yards out.

$ python -m app.cli generate --prompt "With four seconds remaining on the game clock," --max-tokens 60 --num-snippets 1 --seed 42
[Snippet 1] With four seconds remaining on the game clock, the score tied, the point guard brought the ball across half-court against tight full-court pressure and attacked a high ball screen.

$ python -m app.cli generate --prompt "Six runs needed off the final over," --max-tokens 60 --num-snippets 1 --seed 42
[Snippet 1] Six runs needed off the final over, the express bowler executed wide yorkers outside the tramline, conceding only singles while building pressure on the crease.

$ python -m app.cli interactive --checkpoint data/sports_gpt.pt --max-tokens 45
Interactive Sports Play-by-Play Generator
Tip: Type a game scenario prompt (e.g. "Ninety minutes on the clock,") or command (/greedy, /sample, /temp 0.7, /snippets 2, /max-tokens 60, /eval, /help, quit).
Prompt> /temp 0.6
Temperature set to 0.6.
Prompt> /greedy <With four seconds remaining on the game clock>
[Snippet 1] With four seconds remaining on the game clock and the score tied, the point guard drove past his defender on the right elbow and whipped a crisp skip pass to the open corner shooter.
Prompt> quit
```

## Baseline Evaluation & Limitations

Held-out validation benchmark (`python -m app.cli eval --checkpoint data/sports_gpt.pt`, evaluated deterministically across all `2,893` next-token targets in the 10% validation split with `vocab_size=900`, after 400 training steps):

```text
Evaluation on held-out validation split (2893 tokens, vocab=900):
Model                 Cross-Entropy   Perplexity   Top-1 Acc   Top-5 Acc
------------------------------------------------------------------------
Uniform Random               6.8024       900.00       0.11%       0.56%
Unigram (Laplace)            5.7905       327.19       4.84%      14.55%
Bigram (Laplace)             3.9347        51.15      31.80%      73.70%
SportsGPT (Ours)             0.3817         1.46      90.81%      97.30%
------------------------------------------------------------------------
SportsGPT PPL reduction: +99.55% vs Unigram | +97.14% vs Bigram
```

### Limitations

1. **Corpus Diversity & Cross-Sport Drift:** The built-in 186-article corpus (`28,949` BPE tokens) pairs 16 long-form curated articles with 170 shuffled multi-sport commentary scenarios sharing recurring broadcast clauses. While this lets `SportsGPT` learn clean English grammar rapidly on CPU (yielding `1.46` held-out perplexity), sampling under shared connectors (such as `"and"`) can occasionally drift from one sport into another within a single snippet.
2. **Short Context Window (`block_size=64`):** Inputs exceeding 64 BPE tokens are cropped to the most recent 64 tokens, limiting long-range continuity across multi-paragraph game narratives.
3. **No Factual Scoreboard or Rules Grounding:** `SportsGPT` models token co-occurrence statistics only; it does not track legal game state, player rosters, or score math and will invent fictional match details.
4. **CPU Training Speed:** Training time scales with `--steps`, `--batch-size`, and `--d-model`. Large hyperparameter sweeps on CPU take longer without GPU acceleration.

## Testing

All tests run offline without `.env` or external API keys:

```bash
pytest -v
```

Observed suite result: `9 passed in 34.13s` (`0` skipped, `0` failed, `0` warnings; `test_milestone_1` through `test_milestone_6` in `tests/test_core.py`, plus `test_the_store_keeps_what_you_put_in_it`, `test_model_settings_default_to_a_local_ollama`, and `test_the_command_line_runs` in `tests/test_smoke.py`).

## Configuration

Optional LLM settings for `app/llm.py` live in `.env` (copied from `.env.example`), which is ignored by `.gitignore`. Never commit `.env` or API keys. Core training, evaluation, and play-by-play generation in `app/core.py` run completely offline using PyTorch.
