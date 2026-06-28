"""Benchmark offline token counting throughput by provider.

Compares the three local/offline counting paths over a synthetic DataFrame:

* OpenAI    — ``tiktoken`` (Rust)
* Gemini    — the shared Gemma SentencePiece tokenizer via HF ``tokenizers`` (Rust)
* Anthropic — the ~3.5-chars/token heuristic (native Polars, no tokenizer)

Each provider is warmed up once (so the one-time encoding/tokenizer load is
*not* timed), then the per-row counting is timed over several repeats and the
best run reported. Providers whose tokenizer can't be loaded (missing extra, or
a blocked first-use download) are skipped with the reason.

Run::

    pip install "polars-llm[tokens]"
    python examples/benchmark_tokens.py --rows 50000
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import polars as pl

import polars_llm  # noqa: F401  registers the `.llm` namespace

# A few representative lengths so the mix isn't all tiny or all huge.
_SAMPLES = [
    "ok",
    "Summarise polars in one sentence.",
    "The quick brown fox jumps over the lazy dog, then does it again for emphasis.",
    " ".join(["token"] * 60),
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. " * 4,
]

_PROVIDERS: list[tuple[str, str, Callable[[], pl.Expr]]] = [
    ("OpenAI", "tiktoken", lambda: pl.col("text").llm.openai_tokens(model="gpt-4o")),
    ("Gemini", "Gemma SentencePiece", lambda: pl.col("text").llm.gemini_tokens(model="gemini-2.5-pro")),
    ("Anthropic", "char heuristic", lambda: pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6")),
]


def _make_df(rows: int) -> pl.DataFrame:
    texts = [_SAMPLES[i % len(_SAMPLES)] for i in range(rows)]
    return pl.DataFrame({"text": texts})


def _bench(build: Callable[[], pl.Expr], df: pl.DataFrame, repeats: int) -> tuple[float, int]:
    out = df.select(build().alias("n"))  # warmup: loads the encoding/tokenizer
    total_tokens = int(out["n"].sum())
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        df.select(build().alias("n"))
        best = min(best, time.perf_counter() - start)
    return best, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000, help="number of rows to count (default 50000)")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per provider (best is reported)")
    args = parser.parse_args()

    df = _make_df(args.rows)
    print(f"Counting {args.rows:,} rows, best of {args.repeats} runs (tokenizer load excluded)\n")
    header = f"{'provider':<11} {'method':<22} {'best (ms)':>10} {'rows/sec':>14} {'tokens':>12}"
    print(header)
    print("-" * len(header))

    for name, method, build in _PROVIDERS:
        try:
            best, total_tokens = _bench(build, df, args.repeats)
        except Exception as exc:
            print(f"{name:<11} {method:<22} {'skipped':>10}  ({type(exc).__name__})")
            continue
        rows_per_sec = args.rows / best if best > 0 else float("inf")
        print(f"{name:<11} {method:<22} {best * 1000:>10.1f} {rows_per_sec:>14,.0f} {total_tokens:>12,}")


if __name__ == "__main__":
    main()
