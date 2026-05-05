"""Shared sync / async batch runner used by every provider verb.

Each row is invoked through a LangChain ``Runnable`` (chat model, optionally
wrapped in ``with_structured_output``) or an ``Embeddings`` instance. Errors
are caught per-row so a bad input never poisons the whole DataFrame.
"""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from typing import Any, Callable

import nest_asyncio
import polars as pl
from langchain_core.messages import HumanMessage, SystemMessage

OnError = str  # "null" | "raise"

_CHAT_METADATA_FIELDS: dict[str, Any] = {
    "content": pl.Utf8,
    "elapsed_ms": pl.Float64,
    "error": pl.Utf8,
}

_EMBED_METADATA_FIELDS: dict[str, Any] = {
    "vector": pl.List(pl.Float64),
    "dim": pl.Int64,
    "elapsed_ms": pl.Float64,
    "error": pl.Utf8,
}

_CHAT_METADATA_DTYPE = pl.Struct(_CHAT_METADATA_FIELDS)
_EMBED_METADATA_DTYPE = pl.Struct(_EMBED_METADATA_FIELDS)


def _arun(coro: Any) -> Any:
    nest_asyncio.apply()
    return asyncio.run(coro)


def _hashable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, dict):
        return ("__d__", *sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return ("__l__", *(_hashable(v) for v in value))
    return repr(value)


def _build_messages(prompt: Any, system: Any | None) -> list:
    if prompt is None:
        prompt = ""
    msgs: list = []
    if system is not None and system != "":
        msgs.append(SystemMessage(content=str(system)))
    msgs.append(HumanMessage(content=str(prompt)))
    return msgs


def _coerce_chat_content(response: Any) -> str | dict | None:
    """Turn a LangChain response into either a string or a dict (for structured output)."""
    if response is None:
        return None
    if hasattr(response, "content"):
        return str(response.content)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return str(response)


# ---- chat: sync ----
def _chat_one_sync(
    model: Any,
    prompt: Any,
    system: Any,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            response = model.invoke(_build_messages(prompt, system))
            elapsed_ms = (time.monotonic() - start) * 1000
            return {"content": response, "elapsed_ms": elapsed_ms, "error": None}
        except Exception as exc:
            if attempt < retries:
                wait = backoff * (2**attempt) if backoff > 0 else 0.0
                if wait > 0:
                    time.sleep(wait)
                attempt += 1
                continue
            elapsed_ms = (time.monotonic() - start) * 1000
            return {"content": None, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}


def chat_batch_sync(
    model: Any,
    rows: list[tuple[Any, Any]],
    *,
    retries: int,
    backoff: float,
    cache: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(rows)
    memo: dict[Any, dict[str, Any]] = {}
    for i, (prompt, system) in enumerate(rows):
        key: Any = None
        if cache:
            key = (_hashable(prompt), _hashable(system))
            if key in memo:
                results[i] = memo[key]
                continue
        result = _chat_one_sync(model, prompt, system, retries, backoff)
        if cache:
            memo[key] = result
        results[i] = result
    return [r for r in results if r is not None]


# ---- chat: async ----
async def _chat_one_async(
    model: Any,
    semaphore: asyncio.Semaphore | None,
    prompt: Any,
    system: Any,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    async def _go() -> dict[str, Any]:
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                response = await model.ainvoke(_build_messages(prompt, system))
                elapsed_ms = (time.monotonic() - start) * 1000
                return {"content": response, "elapsed_ms": elapsed_ms, "error": None}
            except Exception as exc:
                if attempt < retries:
                    wait = backoff * (2**attempt) if backoff > 0 else 0.0
                    if wait > 0:
                        await asyncio.sleep(wait)
                    attempt += 1
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                return {"content": None, "elapsed_ms": elapsed_ms, "error": f"{type(exc).__name__}: {exc}"}

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def chat_batch_async(
    model: Any,
    rows: list[tuple[Any, Any]],
    *,
    retries: int,
    backoff: float,
    max_concurrency: int | None,
    cache: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    unique_indices: dict[Any, int] = {}
    order: list[int] = []
    result_index: list[int] = [0] * len(rows)
    for i, (prompt, system) in enumerate(rows):
        if cache:
            key = (_hashable(prompt), _hashable(system))
            if key in unique_indices:
                result_index[i] = unique_indices[key]
                continue
            unique_indices[key] = len(order)
            result_index[i] = len(order)
            order.append(i)
        else:
            result_index[i] = len(order)
            order.append(i)

    tasks = [_chat_one_async(model, semaphore, rows[idx][0], rows[idx][1], retries, backoff) for idx in order]
    unique_results = await asyncio.gather(*tasks)
    return [unique_results[result_index[i]] for i in range(len(rows))]


# ---- chat: results -> Series ----
def _warn_silent_errors(silent_errors: list[str], total: int) -> None:
    if not silent_errors:
        return
    warnings.warn(
        f"polars-llm: {len(silent_errors)}/{total} request(s) failed and "
        f"were replaced with null (first error: {silent_errors[0]}). "
        "Pass with_metadata=True to inspect per-row errors, or "
        "on_error='raise' to surface failures immediately.",
        stacklevel=2,
    )


def chat_results_to_series(
    results: list[dict[str, Any]],
    *,
    with_metadata: bool,
    on_error: OnError,
    structured: bool,
) -> pl.Series:
    if with_metadata:
        rows: list[dict[str, Any]] = []
        for r in results:
            if r["error"] is None:
                content_obj = _coerce_chat_content(r["content"])
                if not (content_obj is None or isinstance(content_obj, str)):
                    content_obj = json.dumps(content_obj, default=str)
                rows.append({"content": content_obj, "elapsed_ms": r["elapsed_ms"], "error": None})
            else:
                rows.append({"content": None, "elapsed_ms": r["elapsed_ms"], "error": r["error"]})
        return pl.Series(rows, dtype=_CHAT_METADATA_DTYPE)

    silent_errors: list[str] = []
    if structured:
        out_rows: list[dict | None] = []
        for r in results:
            if r["error"] is None:
                content_obj = _coerce_chat_content(r["content"])
                if content_obj is not None and not isinstance(content_obj, dict):
                    # The model didn't honour the schema — fall through as None.
                    content_obj = None
                out_rows.append(content_obj)
            elif on_error == "raise":
                raise RuntimeError(r["error"])
            else:
                out_rows.append(None)
                silent_errors.append(r["error"])
        _warn_silent_errors(silent_errors, len(results))
        return pl.Series(out_rows)

    out_text: list[str | None] = []
    for r in results:
        if r["error"] is None:
            content = _coerce_chat_content(r["content"])
            out_text.append(content if (content is None or isinstance(content, str)) else json.dumps(content))
        elif on_error == "raise":
            raise RuntimeError(r["error"])
        else:
            out_text.append(None)
            silent_errors.append(r["error"])
    _warn_silent_errors(silent_errors, len(results))
    return pl.Series(out_text, dtype=pl.Utf8)


# ---- embed: sync ----
def _embed_one_sync(
    model: Any,
    text: Any,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            vector = model.embed_query(str(text) if text is not None else "")
            elapsed_ms = (time.monotonic() - start) * 1000
            vector_list = list(vector)
            return {
                "vector": vector_list,
                "dim": len(vector_list),
                "elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as exc:
            if attempt < retries:
                wait = backoff * (2**attempt) if backoff > 0 else 0.0
                if wait > 0:
                    time.sleep(wait)
                attempt += 1
                continue
            elapsed_ms = (time.monotonic() - start) * 1000
            return {
                "vector": None,
                "dim": 0,
                "elapsed_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _embed_chunk_sync(
    model: Any,
    texts: list[Any],
    retries: int,
    backoff: float,
) -> list[dict[str, Any]]:
    inputs = [str(t) if t is not None else "" for t in texts]
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            vectors = model.embed_documents(inputs)
            elapsed_ms = (time.monotonic() - start) * 1000
            return [{"vector": list(v), "dim": len(v), "elapsed_ms": elapsed_ms, "error": None} for v in vectors]
        except Exception as exc:
            if attempt < retries:
                wait = backoff * (2**attempt) if backoff > 0 else 0.0
                if wait > 0:
                    time.sleep(wait)
                attempt += 1
                continue
            elapsed_ms = (time.monotonic() - start) * 1000
            err = f"{type(exc).__name__}: {exc}"
            return [{"vector": None, "dim": 0, "elapsed_ms": elapsed_ms, "error": err} for _ in inputs]


def _dedupe_indices(texts: list[Any], cache: bool) -> tuple[list[int], list[int]]:
    """Return (order, result_index) so that unique texts are at positions
    `[texts[i] for i in order]` and `result_index[i]` maps row i to its slot
    in that unique list. When `cache=False`, returns the identity mapping."""
    if not cache:
        return list(range(len(texts))), list(range(len(texts)))
    unique_indices: dict[Any, int] = {}
    order: list[int] = []
    result_index: list[int] = [0] * len(texts)
    for i, text in enumerate(texts):
        key = _hashable(text)
        if key in unique_indices:
            result_index[i] = unique_indices[key]
            continue
        unique_indices[key] = len(order)
        result_index[i] = len(order)
        order.append(i)
    return order, result_index


def embed_batch_sync(
    model: Any,
    texts: list[Any],
    *,
    retries: int,
    backoff: float,
    cache: bool,
    chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("polars-llm: `chunk_size` must be >= 1")
        order, result_index = _dedupe_indices(texts, cache)
        unique_texts = [texts[idx] for idx in order]
        unique_results: list[dict[str, Any]] = []
        for start in range(0, len(unique_texts), chunk_size):
            chunk = unique_texts[start : start + chunk_size]
            unique_results.extend(_embed_chunk_sync(model, chunk, retries, backoff))
        return [unique_results[result_index[i]] for i in range(len(texts))]

    results: list[dict[str, Any] | None] = [None] * len(texts)
    memo: dict[Any, dict[str, Any]] = {}
    for i, text in enumerate(texts):
        key: Any = None
        if cache:
            key = _hashable(text)
            if key in memo:
                results[i] = memo[key]
                continue
        result = _embed_one_sync(model, text, retries, backoff)
        if cache:
            memo[key] = result
        results[i] = result
    return [r for r in results if r is not None]


# ---- embed: async ----
async def _embed_one_async(
    model: Any,
    semaphore: asyncio.Semaphore | None,
    text: Any,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    async def _go() -> dict[str, Any]:
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                vector = await model.aembed_query(str(text) if text is not None else "")
                elapsed_ms = (time.monotonic() - start) * 1000
                vector_list = list(vector)
                return {
                    "vector": vector_list,
                    "dim": len(vector_list),
                    "elapsed_ms": elapsed_ms,
                    "error": None,
                }
            except Exception as exc:
                if attempt < retries:
                    wait = backoff * (2**attempt) if backoff > 0 else 0.0
                    if wait > 0:
                        await asyncio.sleep(wait)
                    attempt += 1
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                return {
                    "vector": None,
                    "dim": 0,
                    "elapsed_ms": elapsed_ms,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def _embed_chunk_async(
    model: Any,
    semaphore: asyncio.Semaphore | None,
    texts: list[Any],
    retries: int,
    backoff: float,
) -> list[dict[str, Any]]:
    inputs = [str(t) if t is not None else "" for t in texts]

    async def _go() -> list[dict[str, Any]]:
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                vectors = await model.aembed_documents(inputs)
                elapsed_ms = (time.monotonic() - start) * 1000
                return [{"vector": list(v), "dim": len(v), "elapsed_ms": elapsed_ms, "error": None} for v in vectors]
            except Exception as exc:
                if attempt < retries:
                    wait = backoff * (2**attempt) if backoff > 0 else 0.0
                    if wait > 0:
                        await asyncio.sleep(wait)
                    attempt += 1
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                err = f"{type(exc).__name__}: {exc}"
                return [{"vector": None, "dim": 0, "elapsed_ms": elapsed_ms, "error": err} for _ in inputs]

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def embed_batch_async(
    model: Any,
    texts: list[Any],
    *,
    retries: int,
    backoff: float,
    max_concurrency: int | None,
    cache: bool,
    chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("polars-llm: `chunk_size` must be >= 1")
        order, result_index = _dedupe_indices(texts, cache)
        unique_texts = [texts[idx] for idx in order]
        chunks = [unique_texts[i : i + chunk_size] for i in range(0, len(unique_texts), chunk_size)]
        tasks = [_embed_chunk_async(model, semaphore, chunk, retries, backoff) for chunk in chunks]
        chunk_results = await asyncio.gather(*tasks)
        unique_results = [r for chunk in chunk_results for r in chunk]
        return [unique_results[result_index[i]] for i in range(len(texts))]

    unique_indices: dict[Any, int] = {}
    order: list[int] = []
    result_index: list[int] = [0] * len(texts)
    for i, text in enumerate(texts):
        if cache:
            key = _hashable(text)
            if key in unique_indices:
                result_index[i] = unique_indices[key]
                continue
            unique_indices[key] = len(order)
            result_index[i] = len(order)
            order.append(i)
        else:
            result_index[i] = len(order)
            order.append(i)

    tasks = [_embed_one_async(model, semaphore, texts[idx], retries, backoff) for idx in order]
    unique_results = await asyncio.gather(*tasks)
    return [unique_results[result_index[i]] for i in range(len(texts))]


# ---- embed: results -> Series ----
def embed_results_to_series(
    results: list[dict[str, Any]],
    *,
    with_metadata: bool,
    on_error: OnError,
) -> pl.Series:
    if with_metadata:
        return pl.Series(results, dtype=_EMBED_METADATA_DTYPE)

    silent_errors: list[str] = []
    out: list[list[float] | None] = []
    for r in results:
        if r["error"] is None:
            out.append(r["vector"])
        elif on_error == "raise":
            raise RuntimeError(r["error"])
        else:
            out.append(None)
            silent_errors.append(r["error"])
    _warn_silent_errors(silent_errors, len(results))
    return pl.Series(out, dtype=pl.List(pl.Float64))


# ---- map_batches helpers ----
def chat_map_batches(
    input_struct: pl.Expr,
    runner: Callable[[list[tuple[Any, Any]]], list[dict[str, Any]]],
    *,
    with_metadata: bool,
    on_error: OnError,
    structured: bool,
) -> pl.Expr:
    return_dtype = _CHAT_METADATA_DTYPE if with_metadata else (pl.Object if structured else pl.Utf8)

    def _batch(s: pl.Series) -> pl.Series:
        prompts = s.struct.field("prompt").to_list()
        systems = s.struct.field("system").to_list()
        rows = list(zip(prompts, systems))
        results = runner(rows)
        return chat_results_to_series(
            results,
            with_metadata=with_metadata,
            on_error=on_error,
            structured=structured,
        )

    if structured and not with_metadata:
        # Structured output dtype is only known once we see a result; let polars
        # infer from the returned Series.
        return input_struct.map_batches(_batch)
    return input_struct.map_batches(_batch, return_dtype=return_dtype)


def embed_map_batches(
    text_expr: pl.Expr,
    runner: Callable[[list[Any]], list[dict[str, Any]]],
    *,
    with_metadata: bool,
    on_error: OnError,
) -> pl.Expr:
    return_dtype = _EMBED_METADATA_DTYPE if with_metadata else pl.List(pl.Float64)

    def _batch(s: pl.Series) -> pl.Series:
        texts = s.to_list()
        results = runner(texts)
        return embed_results_to_series(
            results,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    return text_expr.map_batches(_batch, return_dtype=return_dtype)
