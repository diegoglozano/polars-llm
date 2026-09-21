"""The ``.llm`` Polars expression namespace.

Importing :mod:`polars_llm` registers the namespace, after which any Polars
expression gains a ``.llm`` accessor with provider-agnostic chat and embedding
verbs, convenience verbs for OpenAI, Anthropic, and Gemini, TypeSafe System
One decisions, and async variants.

Vector columns produced by the embedding verbs additionally gain a
``cosine`` helper that lowers to native Polars arithmetic (no API call).

Provider SDKs are optional extras: install ``polars-llm[openai]``,
``polars-llm[anthropic]``, ``polars-llm[gemini]``, or ``polars-llm[all]``.
TypeSafe support is available through ``polars-llm[typesafe]`` on Python 3.10+.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Mapping
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
from ._typesafe import typesafe_batch_async, typesafe_batch_sync, typesafe_map_batches

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
TypeSafeClient: Any = None
AsyncTypeSafeClient: Any = None

with contextlib.suppress(ImportError):  # pragma: no cover - import guard
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

with contextlib.suppress(ImportError):  # pragma: no cover
    from langchain_anthropic import ChatAnthropic

with contextlib.suppress(ImportError):  # pragma: no cover
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

with contextlib.suppress(ImportError):  # pragma: no cover
    _typesafe_sdk = importlib.import_module("typesafe_sdk")
    AsyncTypeSafeClient = _typesafe_sdk.AsyncTypeSafeClient
    TypeSafeClient = _typesafe_sdk.TypeSafeClient


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
        raise ValueError(f"polars-llm: `model=` is required for `.llm.{provider}` when no `client` is provided.")
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
            f"polars-llm: `model=` is required for `.llm.{provider}_embed` when no `client` is provided.",
        )
    if provider == "openai":
        cls = _require("openai", OpenAIEmbeddings, "openai")
    elif provider == "gemini":
        cls = _require("gemini", GoogleGenerativeAIEmbeddings, "gemini")
    else:  # pragma: no cover
        raise ValueError(f"unknown embedding provider: {provider}")
    return cls(model=model, **model_kwargs)


@pl.api.register_expr_namespace("llm")
class Llm:
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
        chunk_size: int | None,
        dim: int | None,
        with_metadata: bool,
        on_error: OnError,
    ) -> pl.Expr:
        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return embed_batch_sync(
                embedder,
                texts,
                retries=retries,
                backoff=backoff,
                cache=cache,
                chunk_size=chunk_size,
            )

        return embed_map_batches(
            self._prompt,
            runner,
            with_metadata=with_metadata,
            on_error=on_error,
            dim=dim,
        )

    def _aembed(
        self,
        embedder: Any,
        *,
        retries: int,
        backoff: float,
        max_concurrency: int | None,
        cache: bool,
        chunk_size: int | None,
        dim: int | None,
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
                    chunk_size=chunk_size,
                ),
            )

        return embed_map_batches(
            self._prompt,
            runner,
            with_metadata=with_metadata,
            on_error=on_error,
            dim=dim,
        )

    # ---- internal TypeSafe dispatch ----
    def _typesafe(
        self,
        *,
        questions: Mapping[str, Any],
        model: str | None,
        client: Any,
        retries: int,
        backoff: float,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
        client_kwargs: dict[str, Any],
    ) -> pl.Expr:
        client_cls: Any = None if client is not None else _require("typesafe", TypeSafeClient, "typesafe")

        def runner(states: list[Any]) -> list[dict[str, Any]]:
            active_client = client if client is not None else client_cls(**client_kwargs)
            try:
                return typesafe_batch_sync(
                    active_client,
                    states,
                    questions=questions,
                    model=model,
                    retries=retries,
                    backoff=backoff,
                    cache=cache,
                )
            finally:
                if client is None:
                    active_client.close()

        return typesafe_map_batches(
            self._prompt,
            runner,
            questions=questions,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def _atypesafe(
        self,
        *,
        questions: Mapping[str, Any],
        model: str | None,
        client: Any,
        retries: int,
        backoff: float,
        max_concurrency: int | None,
        cache: bool,
        with_metadata: bool,
        on_error: OnError,
        client_kwargs: dict[str, Any],
    ) -> pl.Expr:
        client_cls: Any = None if client is not None else _require("typesafe", AsyncTypeSafeClient, "typesafe")

        async def run(states: list[Any]) -> list[dict[str, Any]]:
            active_client = client if client is not None else client_cls(**client_kwargs)
            try:
                return await typesafe_batch_async(
                    active_client,
                    states,
                    questions=questions,
                    model=model,
                    retries=retries,
                    backoff=backoff,
                    max_concurrency=max_concurrency,
                    cache=cache,
                )
            finally:
                if client is None:
                    await active_client.aclose()

        def runner(states: list[Any]) -> list[dict[str, Any]]:
            return _arun(run(states))

        return typesafe_map_batches(
            self._prompt,
            runner,
            questions=questions,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ============================================================
    # Public chat verbs
    # ============================================================

    # ---- Provider-agnostic ----
    def chat(
        self,
        *,
        client: Any,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
    ) -> pl.Expr:
        """Run chat completions with any LangChain-compatible client.

        ``client`` must provide ``invoke``. When ``schema`` is supplied it
        must also provide ``with_structured_output``. Use this method for
        providers without a dedicated convenience verb, or for custom and
        preconfigured chat clients.
        """
        return self._chat(
            client,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def achat(
        self,
        *,
        client: Any,
        system: str | pl.Expr | None = None,
        schema: Any | None = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
    ) -> pl.Expr:
        """Run chat completions concurrently with any compatible client.

        ``client`` must provide ``ainvoke``. When ``schema`` is supplied it
        must also provide ``with_structured_output``.
        """
        return self._achat(
            client,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

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
        return self.chat(
            client=chat,
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
        return self.achat(
            client=chat,
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
        return self.chat(
            client=chat,
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
        return self.achat(
            client=chat,
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
        return self.chat(
            client=chat,
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
        return self.achat(
            client=chat,
            system=system,
            schema=schema,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ---- TypeSafe System One ----
    def typesafe(
        self,
        *,
        questions: Mapping[str, Any],
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **client_kwargs: Any,
    ) -> pl.Expr:
        """Evaluate TypeSafe Choice, Score, and Noul questions per row.

        The expression is the TypeSafe ``state``. ``questions`` may contain
        ``typesafe_sdk`` question objects or raw question dictionaries. By
        default the result is a nested Struct containing the named answers;
        ``with_metadata=True`` also includes model, token usage, timing, and
        per-row errors. If ``client`` is omitted, remaining keyword arguments
        are forwarded to ``typesafe_sdk.TypeSafeClient``.
        """
        return self._typesafe(
            questions=questions,
            model=model,
            client=client,
            retries=retries,
            backoff=backoff,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
            client_kwargs=client_kwargs,
        )

    def atypesafe(
        self,
        *,
        questions: Mapping[str, Any],
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **client_kwargs: Any,
    ) -> pl.Expr:
        """Evaluate TypeSafe questions concurrently across each batch.

        Uses ``typesafe_sdk.AsyncTypeSafeClient`` unless an async ``client``
        is supplied. ``max_concurrency`` caps in-flight row evaluations.
        """
        return self._atypesafe(
            questions=questions,
            model=model,
            client=client,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            with_metadata=with_metadata,
            on_error=on_error,
            client_kwargs=client_kwargs,
        )

    # ============================================================
    # Public embed verbs
    # ============================================================

    # ---- Provider-agnostic ----
    def embed(
        self,
        *,
        client: Any,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
    ) -> pl.Expr:
        """Compute embeddings with any LangChain-compatible client.

        ``client`` must provide ``embed_query`` and, when ``chunk_size`` is
        supplied, ``embed_documents``.
        """
        return self._embed(
            client,
            retries=retries,
            backoff=backoff,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def aembed(
        self,
        *,
        client: Any,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
    ) -> pl.Expr:
        """Compute embeddings concurrently with any compatible client.

        ``client`` must provide ``aembed_query`` and, when ``chunk_size`` is
        supplied, ``aembed_documents``.
        """
        return self._aembed(
            client,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    def openai_embed(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute OpenAI embeddings per row, sync.

        Pass ``chunk_size=N`` to batch ``N`` rows into a single
        ``embed_documents`` call (cheaper / faster for corpus-style embedding).
        Pass ``dim=N`` to return ``Array(Float64, N)`` instead of the default
        ``List(Float64)`` (catches dim drift, plays nicely with vector libs).
        """
        embedder = _make_embed("openai", model, client, model_kwargs)
        return self.embed(
            client=embedder,
            retries=retries,
            backoff=backoff,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
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
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute OpenAI embeddings concurrently across the batch.

        Pass ``chunk_size=N`` to batch ``N`` rows per ``aembed_documents``
        call; ``max_concurrency`` then caps in-flight chunk calls. Pass
        ``dim=N`` to return ``Array(Float64, N)`` instead of ``List(Float64)``.
        """
        embedder = _make_embed("openai", model, client, model_kwargs)
        return self.aembed(
            client=embedder,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
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
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute Gemini embeddings per row, sync.

        Pass ``chunk_size=N`` to batch ``N`` rows into a single
        ``embed_documents`` call. Pass ``dim=N`` to return
        ``Array(Float64, N)`` instead of ``List(Float64)``.
        """
        embedder = _make_embed("gemini", model, client, model_kwargs)
        return self.embed(
            client=embedder,
            retries=retries,
            backoff=backoff,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
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
        chunk_size: int | None = None,
        dim: int | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Compute Gemini embeddings concurrently across the batch.

        Pass ``chunk_size=N`` to batch ``N`` rows per ``aembed_documents``
        call; ``max_concurrency`` then caps in-flight chunk calls. Pass
        ``dim=N`` to return ``Array(Float64, N)`` instead of ``List(Float64)``.
        """
        embedder = _make_embed("gemini", model, client, model_kwargs)
        return self.aembed(
            client=embedder,
            retries=retries,
            backoff=backoff,
            max_concurrency=max_concurrency,
            cache=cache,
            chunk_size=chunk_size,
            dim=dim,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    # ============================================================
    # Vector helpers (no provider call)
    # ============================================================

    def cosine(self, other: pl.Expr | pl.Series | list[float] | tuple[float, ...]) -> pl.Expr:
        """Cosine similarity between this vector column and ``other``.

        Accepts both ``Array(Float64, dim)`` and ``List(Float64)`` inputs;
        they are cast to ``List`` internally so the math is uniform. ``other``
        may be a ``pl.Expr`` (e.g. ``pl.col("vector_b")``), a ``pl.Series``,
        or a literal Python list/tuple of floats (broadcast against every
        row). Returns a ``Float64`` expression.

        Lowers to native Polars arithmetic — no API call is made. Rows where
        either vector is null produce ``null``; rows where either vector is
        all-zero produce ``NaN`` (0/0).
        """
        list_dtype = pl.List(pl.Float64)
        a = self._prompt.cast(list_dtype)
        if isinstance(other, pl.Expr):
            b: pl.Expr = other.cast(list_dtype)
        elif isinstance(other, pl.Series):
            b = pl.lit(other).cast(list_dtype)
        elif isinstance(other, list | tuple):
            b = pl.lit(pl.Series("", [list(other)], dtype=list_dtype))
        else:
            raise TypeError(
                f"polars-llm: `cosine` expects a pl.Expr, pl.Series, or list of floats; got {type(other).__name__}",
            )
        dot = (a * b).list.sum()
        norm_a = (a * a).list.sum().sqrt()
        norm_b = (b * b).list.sum().sqrt()
        return dot / (norm_a * norm_b)
