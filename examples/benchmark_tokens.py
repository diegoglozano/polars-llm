"""Benchmark offline token-counting throughput by provider and engine.

For each provider it times the offline counting verb twice — once on the native
Rust accelerator (``polars-llm-accel``, an in-engine Polars expression) and once
on the pure-Python UDF fallback — and reports the speedup:

* OpenAI    — tiktoken-rs (native) vs the tiktoken UDF
* Gemini    — the Gemma tokenizer (native) vs the HF ``tokenizers`` UDF
* Anthropic — the ~3.5-chars/token heuristic (already native Polars; no UDF)

Each path is warmed up once (so the one-time tokenizer load is *not* timed), then
timed over several repeats; the best run is reported. Paths whose tokenizer can't
load (missing extra, or a blocked first-use download) are skipped with the reason.

Run::

    pip install "polars-llm[tokens]"     # UDF paths
    pip install polars-llm-accel      # native paths (or build the accel/ crate)
    python examples/benchmark_tokens.py --rows 50000
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import polars as pl

import polars_llm  # noqa: F401  registers the `.llm` namespace
from polars_llm import _tokens as _t

_SAMPLES = [
    "ok",
    "Summarise polars in one sentence.",
    "The quick brown fox jumps over the lazy dog, then does it again for emphasis.",
    " ".join(["token"] * 60),
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. " * 4,
]

# (label, builds a fresh counting expr). Anthropic has no UDF/native split.
_PROVIDERS: list[tuple[str, Callable[[], pl.Expr]]] = [
    ("OpenAI", lambda: pl.col("text").llm.openai_tokens(model="gpt-4o")),
    ("Gemini", lambda: pl.col("text").llm.gemini_tokens(model="gemini-2.5-pro")),
    ("Anthropic (heuristic)", lambda: pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6")),
]


def _make_df(rows: int) -> pl.DataFrame:
    return pl.DataFrame({"text": [_SAMPLES[i % len(_SAMPLES)] for i in range(rows)]})


def _best_ms(build: Callable[[], pl.Expr], df: pl.DataFrame, repeats: int) -> float | None:
    try:
        df.select(build().alias("n"))  # warmup (loads tokenizer); also surfaces failures
    except Exception:
        return None
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        df.select(build().alias("n"))
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    df = _make_df(args.rows)
    have_accel = _t._ACCEL is not None
    print(
        f"Counting {args.rows:,} rows, best of {args.repeats} (tokenizer load excluded); accel={'yes' if have_accel else 'no'}\n"
    )
    header = f"{'provider':<22} {'native (ms)':>12} {'UDF (ms)':>12} {'speedup':>9}  {'rows/sec (best)':>16}"
    print(header)
    print("-" * len(header))

    for name, build in _PROVIDERS:
        # Native run (verb uses the accelerator when present).
        native = _best_ms(build, df, args.repeats) if have_accel else None
        # UDF run: force the pure-Python fallback.
        saved, _t._ACCEL = _t._ACCEL, None
        try:
            udf = _best_ms(build, df, args.repeats)
        finally:
            _t._ACCEL = saved

        best = min([t for t in (native, udf) if t is not None], default=None)
        if best is None:
            print(f"{name:<22} {'skipped':>12} {'skipped':>12} {'-':>9}  {'-':>16}")
            continue
        speedup = f"{udf / native:.1f}x" if (native and udf) else "-"
        print(
            f"{name:<22} {('-' if native is None else f'{native:.1f}'):>12} "
            f"{('-' if udf is None else f'{udf:.1f}'):>12} {speedup:>9}  {args.rows / (best / 1000):>16,.0f}"
        )


if __name__ == "__main__":
    main()
