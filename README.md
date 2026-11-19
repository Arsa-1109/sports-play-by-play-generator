# Sports Play-by-Play Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/pzuatu3vjh/badge.svg)](https://ask.thili.ai/b/pzuatu3vjh)

This project delivers a command-line assistant for sports fans, generating dynamic commentary snippets for fictional game scenarios. You will build a GPT model from scratch, applying byte-pair tokenization, embeddings, and causal self-attention to train it on real sports news articles, ultimately producing coherent and engaging sports commentary.

## What you are building

**Goal.** You will build a command-line assistant that generates short, dynamic sports commentary snippets for fictional game scenarios, trained on real sports news articles.

**Done when.** The project is complete when the command-line interface successfully takes a user prompt and generates coherent, short sports commentary snippets using a GPT model trained from scratch.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Gather a small corpus of sports commentary or news articles.**  
      Hint: Data Loading & Preprocessing
- [ ] **2. Implement the byte-pair tokenizer and create embeddings from the sports text.**  
      Hint: Byte-Pair Tokenizer; Token Embeddings
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Causal Self-Attention; Transformer Block
- [ ] **4. Train the GPT model using the sports text data.**  
      Hint: Training Loop; Loss Calculation
- [ ] **5. Develop a text generation function to produce short commentary snippets.**  
      Hint: Sampling; Greedy Decoding
- [ ] **6. Create a CLI where users can input a prompt and receive generated commentary.**  
      Hint: User Input Handling

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify the desired length or number of commentary snippets directly within the CLI.
- Experiment with different model hyperparameters or a slightly expanded corpus to improve the quality and variety of generated commentary.
