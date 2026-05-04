"""The ``.ai`` Polars expression namespace.

Importing :mod:`polars_llm` registers the namespace, after which any Polars
expression that resolves to a string column gains an ``.ai`` accessor with one
verb per provider (``openai``, ``anthropic``, ``gemini``) plus async variants
(``aopenai``, ``aanthropic``, ``agemini``) and embedding variants
(``openai_embed`` / ``gemini_embed`` and their async counterparts).

Provider SDKs are optional extras: install ``polars-llm[openai]``,
``polars-llm[anthropic]``, ``polars-llm[gemini]``, or ``polars-llm[all]``.
"""

from __future__ import annotations

import contextlib
from typing import Any

import polars as pl

from ._runtime import (
    OnError,
    _arun,
    chat_batch_async,
    chat_batch_sync,
    chat_map_batches,
    embed_batch_async,
    embed_batch_sync,
    embed_map_batches,
)

# ---- Optional provider imports ----
# Each name is set to None up front so it's a stable module attribute even when
# the corresponding extra isn't installed; the try/except below rebinds them to
# the real LangChain classes when available. Type as Any so monkeypatching in
# tests doesn't fight a narrower static type.
ChatOpenAI: Any = None
OpenAIEmbeddings: Any = None
ChatAnthropic: Any = None
ChatGoogleGenerativeAI: Any = None
GoogleGenerativeAIEmbeddings: Any = None

with contextlib.suppress(ImportError):  # pragma: no cover - import guard
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

with contextlib.suppress(ImportError):  # pragma: no cover
    from langchain_anthropic import ChatAnthropic

with contextlib.suppress(ImportError):  # pragma: no cover
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )


def _require(provider: str, cls: Any, extra: str) -> Any:
    if cls is None:
        raise ImportError(
            f"polars-llm: the {provider!r} provider requires the optional "
            f"`{extra}` extra. Install it with `pip install polars-llm[{extra}]`.",
        )
    return cls


def _make_chat(
    provider: str,
    model: str | None,
    client: Any,
    model_kwargs: dict[str, Any],
) -> Any:
    if client is not None:
        return client
    if model is None:
        raise ValueError(f"polars-llm: `model=` is required for `.ai.{provider}` when no `client` is provided.")
    if provider == "openai":
        cls = _require("openai", ChatOpenAI, "openai")
    elif provider == "anthropic":
        cls = _require("anthropic", ChatAnthropic, "anthropic")
    elif provider == "gemini":
        cls = _require("gemini", ChatGoogleGenerativeAI, "gemini")
    else:  # pragma: no cover - guarded at call site
        raise ValueError(f"unknown provider: {provider}")
    return cls(model=model, **model_kwargs)


def _make_embed(
    provider: str,
    model: str | None,
    client: Any,
    model_kwargs: dict[str, Any],
) -> Any:
    if client is not None:
        return client
    if model is None:
        raise ValueError(
            f"polars-llm: `model=` is required for `.ai.{provider}_embed` when no `client` is provided.",
        )
    if provider == "openai":
        cls = _require("openai", OpenAIEmbeddings, "openai")
    elif provider == "gemini":
        cls = _require("gemini", GoogleGenerativeAIEmbeddings, "gemini")
    else:  # pragma: no cover
        raise ValueError(f"unknown embedding provider: {provider}")
    return cls(model=model, **model_kwargs)


@pl.api.register_expr_namespace("ai")
class Ai:
    """Expression namespace for calling LLMs and embedding models per row."""

    def __init__(self, prompt: pl.Expr) -> None:
        self._prompt = prompt

    # ---- input shaping ----
    def _input_struct(self, system: str | pl.Expr | None) -> pl.Expr:
        if system is None:
            sys_expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
        elif isinstance(system, pl.Expr):
            sys_expr = system.cast(pl.Utf8)
        else:
            sys_expr = pl.lit(str(system))
        return pl.struct(self._prompt.alias("prompt"), sys_expr.alias("system"))

    # ---- internal chat dispatch ----
    def _chat(
        self,
        chat: Any,
        *,
        system: str | pl.Expr | None,
        schema: Any | None,
        retries: int,
        backoff: float,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
    ) -> pl.Expr:
        if schema is not None:
            chat = chat.with_structured_output(schema)

        def runner(rows: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
            return chat_batch_sync(chat, rows, retries=retries, backoff=backoff, cache=cache)

        return chat_map_batches(
            self._input_struct(system),
            runner,
            with_metadata=with_metadata,
            on_error=on_error,
            structured=schema is not None,
        )

    def _achat(
        self,
        chat: Any,
        *,
        system: str | pl.Expr | None,
        schema: Any | None,
        retries: int,
        backoff: float,
        max_concurrency: int | None,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
    ) -> pl.Expr:
        if schema is not None:
            chat = chat.with_structured_output(schema)

        def runner(rows: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
            return _arun(
                chat_batch_async(
                    chat,
                    rows,
                    retries=retries,
                    backoff=backoff,
                    max_concurrency=max_concurrency,
                    cache=cache,
                ),
            )

        return chat_map_batches(
            self._input_struct(system),
            runner,
            with_metadata=with_metadata,
            on_error=on_error,
            structured=schema is not None,
        )

    # ---- internal embed dispatch ----
    def _embed(
        self,
        embedder: Any,
        *,
        retries: int,
        backoff: float,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
    ) -> pl.Expr:
        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return embed_batch_sync(embedder, texts, retries=retries, backoff=backoff, cache=cache)

        return embed_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    def _aembed(
        self,
        embedder: Any,
        *,
        retries: int,
        backoff: float,
        max_concurrency: int | None,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
    ) -> pl.Expr:
        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return _arun(
                embed_batch_async(
                    embedder,
                    texts,
                    retries=retries,
                    backoff=backoff,
                    max_concurrency=max_concurrency,
                    cache=cache,
                ),
            )

        return embed_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    # ============================================================
    # Public chat verbs
    # ============================================================

    # ---- OpenAI ----
    def openai(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run an OpenAI chat completion per row, sync."""
        chat = _make_chat("openai", model, client, model_kwargs)
        return self._chat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def aopenai(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run OpenAI chat completions concurrently across the batch."""
        chat = _make_chat("openai", model, client, model_kwargs)
        return self._achat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ---- Anthropic ----
    def anthropic(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run an Anthropic chat completion per row, sync."""
        chat = _make_chat("anthropic", model, client, model_kwargs)
        return self._chat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def aanthropic(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run Anthropic chat completions concurrently across the batch."""
        chat = _make_chat("anthropic", model, client, model_kwargs)
        return self._achat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ---- Gemini ----
    def gemini(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run a Gemini chat completion per row, sync."""
        chat = _make_chat("gemini", model, client, model_kwargs)
        return self._chat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def agemini(
        self,
        *,
        model: str | None = None,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Run Gemini chat completions concurrently across the batch."""
        chat = _make_chat("gemini", model, client, model_kwargs)
        return self._achat(
            chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ============================================================
    # Public embed verbs
    # ============================================================

    def openai_embed(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute OpenAI embeddings per row, sync."""
        embedder = _make_embed("openai", model, client, model_kwargs)
        return self._embed(
            embedder,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def aopenai_embed(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute OpenAI embeddings concurrently across the batch."""
        embedder = _make_embed("openai", model, client, model_kwargs)
        return self._aembed(
            embedder,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def gemini_embed(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute Gemini embeddings per row, sync."""
        embedder = _make_embed("gemini", model, client, model_kwargs)
        return self._embed(
            embedder,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def agemini_embed(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute Gemini embeddings concurrently across the batch."""
        embedder = _make_embed("gemini", model, client, model_kwargs)
        return self._aembed(
            embedder,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )
