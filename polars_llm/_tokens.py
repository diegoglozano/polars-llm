"""Token counting and cost estimation behind the ``.llm`` namespace.

Three methodologies, one per provider — see the table below. The fast paths
lean on tokenizer libraries that are *already* Rust-backed (``tiktoken``,
HuggingFace ``tokenizers``), so this stays a pure-Python package while still
counting at native speed.

============  =========================  ==========================================
Provider      Offline & exact?           How
============  =========================  ==========================================
OpenAI        yes (local, fast)          ``tiktoken`` — ``encoding_for_model``
Gemini        yes (local, fast)          shared **Gemma** SentencePiece tokenizer
                                         (262,144 vocab) via HF ``tokenizers``;
                                         counts match Google's API, no network
Anthropic     no public tokenizer        offline = Anthropic's documented heuristic
                                         (~1 token ≈ 3.5 chars, an *estimate*);
                                         exact = the ``count_tokens`` API
============  =========================  ==========================================

The tokenizer backends are an optional extra::

    pip install polars-llm[tokenizers]
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from ._runtime import OnError, _dedupe_indices, _warn_silent_errors

# ---- optional tokenizer backends ----
# Set to None up front so each is a stable module attribute even when the
# `tokenizers` extra isn't installed (and so tests can monkeypatch them).
_TIKTOKEN: Any = None
with contextlib.suppress(ImportError):  # pragma: no cover - import guard
    import tiktoken as _TIKTOKEN

_TOKENIZERS: Any = None
with contextlib.suppress(ImportError):  # pragma: no cover - import guard
    from tokenizers import Tokenizer as _TOKENIZERS

# Gemini and the open Gemma models share one SentencePiece tokenizer. The
# canonical repo is gated on Hugging Face; override the source with a local
# file (`tokenizer_path=` / this env var) or point at an un-gated mirror repo.
GEMMA_TOKENIZER_REPO = os.environ.get("POLARS_LLM_GEMMA_REPO", "google/gemma-2-2b")

_TOKENS_METADATA_FIELDS: dict[str, Any] = {
    "tokens": pl.Int64,
    "elapsed_ms": pl.Float64,
    "error": pl.Utf8,
}
_TOKENS_METADATA_DTYPE = pl.Struct(_TOKENS_METADATA_FIELDS)


# ============================================================
# Optional-extra guards
# ============================================================
def _require_tiktoken() -> Any:
    if _TIKTOKEN is None:
        raise ImportError(
            "polars-llm: OpenAI token counting requires the optional `tokenizers` "
            "extra. Install it with `pip install polars-llm[tokenizers]`.",
        )
    return _TIKTOKEN


def _require_tokenizers() -> Any:
    if _TOKENIZERS is None:
        raise ImportError(
            "polars-llm: Gemini (local) token counting requires the optional "
            "`tokenizers` extra. Install it with `pip install polars-llm[tokenizers]`.",
        )
    return _TOKENIZERS


# ============================================================
# OpenAI — local, exact (tiktoken)
# ============================================================
@functools.cache
def _encoding_for(model: str | None) -> Any:
    tk = _require_tiktoken()
    if model:
        try:
            return tk.encoding_for_model(model)
        except KeyError:
            pass  # new / unknown model name — fall back to the latest base encoding
    return tk.get_encoding("o200k_base")


def count_openai(texts: list[Any], *, model: str | None) -> list[int | None]:
    """Token count per row via tiktoken. ``None`` in → ``None`` out, ``""`` → 0."""
    enc = _encoding_for(model)
    nonnull = [i for i, t in enumerate(texts) if t is not None]
    inputs = [str(texts[i]) for i in nonnull]
    # `disallowed_special=()` so user text containing literals like
    # "<|endoftext|>" is counted as plain text instead of raising.
    encoded = enc.encode_batch(inputs, disallowed_special=()) if inputs else []
    out: list[int | None] = [None] * len(texts)
    for k, i in enumerate(nonnull):
        out[i] = len(encoded[k])
    return out


# ============================================================
# Gemini — local, exact (shared Gemma SentencePiece tokenizer)
# ============================================================
@functools.cache
def _gemma_tokenizer(tokenizer_path: str | None = None) -> Any:
    tok_cls = _require_tokenizers()
    src = tokenizer_path or os.environ.get("POLARS_LLM_GEMMA_TOKENIZER")
    if src:
        return tok_cls.from_file(src)
    try:
        return tok_cls.from_pretrained(GEMMA_TOKENIZER_REPO)
    except Exception as exc:  # download / auth / network failure
        raise RuntimeError(
            "polars-llm: could not load the Gemma tokenizer for local Gemini token "
            f"counting from {GEMMA_TOKENIZER_REPO!r} ({type(exc).__name__}: {exc}). "
            "Pass a local `tokenizer_path=`, set POLARS_LLM_GEMMA_TOKENIZER / "
            "POLARS_LLM_GEMMA_REPO to an accessible source, or use `exact=True` to "
            "count via the Gemini API instead.",
        ) from exc


def count_gemini(texts: list[Any], *, tokenizer: Any) -> list[int | None]:
    """Token count per row via the Gemma tokenizer. ``None`` in → ``None`` out."""
    nonnull = [i for i, t in enumerate(texts) if t is not None]
    inputs = [str(texts[i]) for i in nonnull]
    encoded = tokenizer.encode_batch(inputs) if inputs else []
    out: list[int | None] = [None] * len(texts)
    for k, i in enumerate(nonnull):
        out[i] = len(encoded[k].ids)
    return out


# ============================================================
# Anthropic — offline heuristic (native Polars, no dependency)
# ============================================================
def anthropic_offline_expr(text_expr: pl.Expr, chars_per_token: float) -> pl.Expr:
    """Anthropic's documented estimate: ~1 token per ``chars_per_token`` chars.

    Lowers to native Polars arithmetic (no API call, no tokenizer). Nulls pass
    through; ``""`` → 0. This is an *estimate* — use ``exact=True`` for an
    API-accurate count.
    """
    if chars_per_token <= 0:
        raise ValueError("polars-llm: `chars_per_token` must be > 0.")
    return (text_expr.str.len_chars().cast(pl.Float64) / chars_per_token).ceil().cast(pl.Int64)


# ============================================================
# Exact counting (network): batching / dedup / retries / concurrency
# ============================================================
def _count_one_sync(counter: Callable[[str], int], text: Any, retries: int, backoff: float) -> dict[str, Any]:
    if text is None:
        return {"tokens": None, "elapsed_ms": 0.0, "error": None}
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            tokens = int(counter(str(text)))
            return {"tokens": tokens, "elapsed_ms": (time.monotonic() - start) * 1000, "error": None}
        except Exception as exc:
            if attempt < retries:
                wait = backoff * (2**attempt) if backoff > 0 else 0.0
                if wait > 0:
                    time.sleep(wait)
                attempt += 1
                continue
            elapsed_ms = (time.monotonic() - start) * 1000
            return {"tokens": None, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}


def count_batch_sync(
    counter: Callable[[str], int],
    texts: list[Any],
    *,
    retries: int,
    backoff: float,
    cache: bool,
) -> list[dict[str, Any]]:
    order, result_index = _dedupe_indices(texts, cache)
    unique = [_count_one_sync(counter, texts[idx], retries, backoff) for idx in order]
    return [unique[result_index[i]] for i in range(len(texts))]


async def _count_one_async(
    acounter: Callable[[str], Any],
    semaphore: asyncio.Semaphore | None,
    text: Any,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    if text is None:
        return {"tokens": None, "elapsed_ms": 0.0, "error": None}

    async def _go() -> dict[str, Any]:
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                tokens = int(await acounter(str(text)))
                return {"tokens": tokens, "elapsed_ms": (time.monotonic() - start) * 1000, "error": None}
            except Exception as exc:
                if attempt < retries:
                    wait = backoff * (2**attempt) if backoff > 0 else 0.0
                    if wait > 0:
                        await asyncio.sleep(wait)
                    attempt += 1
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                return {"tokens": None, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def count_batch_async(
    acounter: Callable[[str], Any],
    texts: list[Any],
    *,
    retries: int,
    backoff: float,
    max_concurrency: int | None,
    cache: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    order, result_index = _dedupe_indices(texts, cache)
    tasks = [_count_one_async(acounter, semaphore, texts[idx], retries, backoff) for idx in order]
    unique = await asyncio.gather(*tasks)
    return [unique[result_index[i]] for i in range(len(texts))]


# ============================================================
# results -> Series + map_batches helper
# ============================================================
def tokens_results_to_series(
    results: list[dict[str, Any]],
    *,
    with_metadata: bool,
    on_error: OnError,
) -> pl.Series:
    if with_metadata:
        return pl.Series(results, dtype=_TOKENS_METADATA_DTYPE)

    silent_errors: list[str] = []
    out: list[int | None] = []
    for r in results:
        if r["error"] is None:
            out.append(r["tokens"])
        elif on_error == "raise":
            raise RuntimeError(r["error"])
        else:
            out.append(None)
            silent_errors.append(r["error"])
    _warn_silent_errors(silent_errors, len(results))
    return pl.Series(out, dtype=pl.Int64)


def tokens_map_batches(
    text_expr: pl.Expr,
    runner: Callable[[list[Any]], list[dict[str, Any]]],
    *,
    with_metadata: bool,
    on_error: OnError,
) -> pl.Expr:
    return_dtype: Any = _TOKENS_METADATA_DTYPE if with_metadata else pl.Int64

    def _batch(s: pl.Series) -> pl.Series:
        results = runner(s.to_list())
        return tokens_results_to_series(results, with_metadata=with_metadata, on_error=on_error)

    return text_expr.map_batches(_batch, return_dtype=return_dtype)


# ============================================================
# Pricing & cost
# ============================================================
@dataclass(frozen=True)
class Price:
    """USD per 1,000,000 tokens, split by direction."""

    input_per_1m: float
    output_per_1m: float


# Approximate published list prices, last reviewed 2026-06. These drift — treat
# them as a convenience default and override per call (`prices=...`) or globally
# by mutating `polars_llm.PRICES`.
PRICES: dict[str, Price] = {
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
    "claude-3-5-haiku": Price(0.80, 4.00),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-opus-4-1": Price(15.00, 75.00),
    "gemini-2.0-flash": Price(0.10, 0.40),
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-pro": Price(1.25, 10.00),
}

_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt", "o1", "o3", "o4", "chatgpt", "text-embedding"),
    "anthropic": ("claude",),
    "gemini": ("gemini", "models/gemini"),
}


def infer_provider(model: str) -> str:
    name = model.lower()
    for provider, prefixes in _PROVIDER_PREFIXES.items():
        if name.startswith(prefixes):
            return provider
    raise ValueError(
        f"polars-llm: could not infer a provider from model {model!r}; pass `provider=` explicitly.",
    )


def _price_for(model: str, prices: dict[str, Price] | None) -> Price:
    table = prices if prices is not None else PRICES
    if model in table:
        return table[model]
    for key, price in table.items():
        if model.startswith(key) or key.startswith(model):
            return price
    raise ValueError(
        f"polars-llm: no price entry for model {model!r}. Pass `prices=` or add it to `polars_llm.PRICES`.",
    )


def price_per_token(model: str, kind: str = "input", prices: dict[str, Price] | None = None) -> float:
    """USD per single token for ``model`` in the given direction (``"input"``/``"output"``)."""
    if kind not in ("input", "output"):
        raise ValueError(f"polars-llm: `kind` must be 'input' or 'output'; got {kind!r}.")
    price = _price_for(model, prices)
    per_1m = price.input_per_1m if kind == "input" else price.output_per_1m
    return per_1m / 1_000_000
