"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
import re
import sys

_PLACEHOLDER_RE = re.compile(r"^<([^<>]+)>$")


def _strip_placeholders(text: str) -> str:
    """Strip accidental angle-bracket placeholder wrappers like `<your prompt>`."""
    cleaned = text.strip()
    match = _PLACEHOLDER_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-tokens", type=int, default=80, help="maximum tokens to generate per snippet (>= 1)")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature (<=0 uses greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="top-k sampling cutoff (0 disables)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus (top-p) sampling threshold in (0, 1]")
    parser.add_argument("--greedy", action="store_true", help="use deterministic greedy decoding")
    parser.add_argument("--num-snippets", type=int, default=1, help="number of commentary snippets to generate (>= 1)")
    parser.add_argument(
        "--checkpoint",
        default="data/sports_gpt.pt",
        help="path to trained SportsGPT checkpoint",
    )
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed for reproducible sampling")


def _validate_gen_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.num_snippets < 1:
        parser.error("--num-snippets must be >= 1")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        parser.error("--top-p must be in (0, 1]")


def _format_eval_report(report: dict) -> str:
    lines = [
        f"Evaluation on held-out validation split ({report['val_eval_tokens']} tokens, vocab={report['vocab_size']}):",
        f"{'Model':<20} {'Cross-Entropy':>14} {'Perplexity':>12} {'Top-1 Acc':>11} {'Top-5 Acc':>11}",
        "-" * 72,
    ]
    for m in report["models"]:
        lines.append(
            f"{m['name']:<20} {m['cross_entropy']:>14.4f} {m['perplexity']:>12.2f} "
            f"{m['top1_acc'] * 100:>10.2f}% {m['top5_acc'] * 100:>10.2f}%"
        )
    lines.append("-" * 72)
    lines.append(
        f"SportsGPT PPL reduction: {report['ppl_reduction_vs_unigram_pct']:+.2f}% vs Unigram | "
        f"{report['ppl_reduction_vs_bigram_pct']:+.2f}% vs Bigram"
    )
    return "\n".join(lines)


def _run_interactive_session(
    args: argparse.Namespace,
    model,
    tokenizer,
    evaluate_fn,
    generate_fn,
) -> None:
    state = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k if args.top_k > 0 else None,
        "top_p": args.top_p,
        "greedy": args.greedy,
        "num_snippets": args.num_snippets,
        "seed": args.seed,
    }
    print("Interactive Sports Play-by-Play Generator")
    print(
        'Tip: Type a game scenario prompt (e.g. "Ninety minutes on the clock,") '
        "or command (/greedy, /sample, /temp 0.7, /snippets 2, /max-tokens 60, /eval, /help, quit)."
    )

    while True:
        try:
            raw_line = input("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = _strip_placeholders(raw_line)
        if not line:
            continue

        lower = line.lower()
        if lower in {"quit", "exit", "/quit", "/exit"}:
            break

        if lower == "/help":
            print(
                "Commands: /greedy [prompt] | /sample [prompt] | /temp <float> [| prompt] | "
                "/snippets <int> | /max-tokens <int> | /eval | quit"
            )
            continue

        if lower == "/eval":
            report = evaluate_fn(checkpoint_path=args.checkpoint, train_if_missing=True)
            print(_format_eval_report(report))
            continue

        # Handle standalone or inline slash commands.
        prompt_text: str | None = None
        if line.startswith("/"):
            parts = line.split(None, 1)
            cmd = parts[0].lower()
            rest = _strip_placeholders(parts[1]) if len(parts) > 1 else ""

            if cmd == "/greedy":
                state["greedy"] = True
                if not rest:
                    print("Mode set to greedy decoding.")
                    continue
                prompt_text = rest
            elif cmd == "/sample":
                state["greedy"] = False
                if not rest:
                    print("Mode set to stochastic sampling.")
                    continue
                prompt_text = rest
            elif cmd == "/temp":
                val_and_prompt = rest.split("|", 1)
                try:
                    temp_val = float(_strip_placeholders(val_and_prompt[0]))
                except ValueError:
                    print("Invalid /temp value. Example: /temp 0.7")
                    continue
                state["temperature"] = temp_val
                state["greedy"] = temp_val <= 0.0
                if len(val_and_prompt) == 1 or not val_and_prompt[1].strip():
                    print(f"Temperature set to {temp_val}.")
                    continue
                prompt_text = _strip_placeholders(val_and_prompt[1])
            elif cmd == "/snippets":
                try:
                    n = int(rest)
                    if n < 1:
                        raise ValueError
                except ValueError:
                    print("Invalid /snippets count. Example: /snippets 2")
                    continue
                state["num_snippets"] = n
                print(f"Snippets per prompt set to {n}.")
                continue
            elif cmd == "/max-tokens":
                try:
                    n = int(rest)
                    if n < 1:
                        raise ValueError
                except ValueError:
                    print("Invalid /max-tokens count. Example: /max-tokens 60")
                    continue
                state["max_tokens"] = n
                print(f"Max tokens per snippet set to {n}.")
                continue
            else:
                print(f"Unknown command '{cmd}'. Type /help for options or enter a plain prompt.")
                continue
        else:
            if lower.startswith("generate "):
                prompt_text = _strip_placeholders(line[len("generate ") :])
            elif lower.startswith("generate:"):
                prompt_text = _strip_placeholders(line[len("generate:") :])
            else:
                prompt_text = line

        snippets = generate_fn(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text or "",
            max_new_tokens=state["max_tokens"],
            temperature=state["temperature"],
            top_k=state["top_k"],
            top_p=state["top_p"],
            greedy=state["greedy"],
            num_snippets=state["num_snippets"],
            seed=state["seed"],
        )
        for idx, snippet in enumerate(snippets, start=1):
            print(f"[Snippet {idx}] {snippet}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description=(
            "Sports fans get a command-line assistant that generates short, dynamic commentary snippets "
            "for fictional game scenarios, trained on real sports news articles."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    train_p = sub.add_parser("train", help="train BPE tokenizer and SportsGPT from scratch")
    train_p.add_argument("--steps", type=int, default=200, help="number of AdamW training steps (>= 1)")
    train_p.add_argument("--vocab-size", type=int, default=400, help="BPE vocabulary size (>= 256)")
    train_p.add_argument("--d-model", type=int, default=96, help="transformer embedding dimension")
    train_p.add_argument("--n-heads", type=int, default=4, help="number of causal attention heads")
    train_p.add_argument("--n-layers", type=int, default=2, help="number of transformer blocks")
    train_p.add_argument("--block-size", type=int, default=64, help="maximum context window length (>= 2)")
    train_p.add_argument("--batch-size", type=int, default=12, help="training batch size (>= 1)")
    train_p.add_argument("--lr", type=float, default=2.5e-3, help="peak learning rate (> 0)")
    train_p.add_argument("--corpus", default=None, help="optional path to custom UTF-8 sports corpus")
    train_p.add_argument(
        "--checkpoint",
        default="data/sports_gpt.pt",
        help="output path for trained checkpoint",
    )
    train_p.add_argument("--seed", type=int, default=42, help="random seed")

    gen_p = sub.add_parser("generate", help="generate play-by-play commentary snippets from a prompt")
    gen_p.add_argument(
        "--prompt",
        default="Ninety minutes on the clock, and the striker",
        help="game scenario prompt to continue",
    )
    _add_generation_args(gen_p)

    eval_p = sub.add_parser("eval", help="benchmark SportsGPT against Uniform, Unigram, and Bigram baselines")
    eval_p.add_argument("--checkpoint", default="data/sports_gpt.pt", help="path to trained SportsGPT checkpoint")
    eval_p.add_argument("--corpus", default=None, help="optional path to custom UTF-8 sports corpus")
    eval_p.add_argument("--steps", type=int, default=200, help="training steps if checkpoint does not yet exist")
    eval_p.add_argument("--json", action="store_true", help="emit machine-readable JSON evaluation metrics")

    inter_p = sub.add_parser("interactive", help="interactive REPL for live commentary generation")
    _add_generation_args(inter_p)

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")
        return

    if args.command == "train":
        if args.steps < 1:
            parser.error("--steps must be >= 1")
        if args.vocab_size < 256:
            parser.error("--vocab-size must be >= 256")
        if args.d_model % args.n_heads != 0:
            parser.error("--d-model must be divisible by --n-heads")
        if args.block_size < 2:
            parser.error("--block-size must be >= 2")
        if args.batch_size < 1:
            parser.error("--batch-size must be >= 1")
        if args.lr <= 0:
            parser.error("--lr must be > 0")
    elif args.command in {"generate", "interactive"}:
        _validate_gen_args(parser, args)
    elif args.command == "eval":
        if args.steps < 1:
            parser.error("--steps must be >= 1")

    from app.core import (
        ensure_trained_checkpoint,
        evaluate_baseline,
        generate_text,
        load_checkpoint,
        train_model,
    )

    try:
        if args.command == "train":
            result = train_model(
                corpus_path=args.corpus,
                checkpoint_path=args.checkpoint,
                vocab_size=args.vocab_size,
                block_size=args.block_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_interval=max(1, args.steps // 2),
                seed=args.seed,
                verbose=True,
            )
            print(
                f"Training complete | tokens={result['num_tokens']} | "
                f"initial_loss={result['initial_loss']:.4f} -> "
                f"final_train_loss={result['final_train_loss']:.4f} | "
                f"final_val_loss={result['final_val_loss']:.4f} | "
                f"saved={result['checkpoint_path']}"
            )
            return

        if args.command == "generate":
            ckpt = ensure_trained_checkpoint(args.checkpoint, verbose=True)
            model, tokenizer, _, _ = load_checkpoint(ckpt)
            clean_prompt = _strip_placeholders(args.prompt)
            snippets = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=clean_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                greedy=args.greedy,
                num_snippets=args.num_snippets,
                seed=args.seed,
            )
            for idx, snippet in enumerate(snippets, start=1):
                print(f"[Snippet {idx}] {snippet}")
            return

        if args.command == "eval":
            report = evaluate_baseline(
                checkpoint_path=args.checkpoint,
                corpus_path=args.corpus,
                steps=args.steps,
                train_if_missing=True,
            )
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(_format_eval_report(report))
            return

        if args.command == "interactive":
            ckpt = ensure_trained_checkpoint(args.checkpoint, verbose=True)
            model, tokenizer, _, _ = load_checkpoint(ckpt)
            _run_interactive_session(args, model, tokenizer, evaluate_baseline, generate_text)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
