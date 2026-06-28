"""The ``.llm`` Polars expression namespace.

Importing :mod:`polars_llm` registers the namespace, after which any Polars
expression that resolves to a string column gains a ``.llm`` accessor with one
verb per provider (``openai``, ``anthropic``, ``gemini``) plus async variants
(``aopenai``, ``aanthropic``, ``agemini``) and embedding variants
(``openai_embed`` / ``gemini_embed`` and their async counterparts).

Vector columns produced by the embedding verbs additionally gain a
``cosine`` helper that lowers to native Polars arithmetic (no API call).

Provider SDKs are optional extras: install ``polars-llm[openai]``,
``polars-llm[anthropic]``, ``polars-llm[gemini]``, or ``polars-llm[all]``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Literal

import polars as pl
from langchain_core.messages import HumanMessage

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
from ._tokens import (
    Price,
    anthropic_offline_expr,
    count_batch_async,
    count_batch_sync,
    gemini_offline_expr,
    infer_provider,
    openai_tokens_expr,
    price_per_token,
    tokens_map_batches,
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
        return self._embed(
            embedder,
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
        return self._aembed(
            embedder,
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
        return self._embed(
            embedder,
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
        return self._aembed(
            embedder,
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
    # Token counting & cost estimation
    # ============================================================

    def openai_tokens(
        self,
        *,
        model: str | None = None,
        with_metadata: bool = False,
        on_error: OnError = "null",
    ) -> pl.Expr:
        """Count OpenAI tokens per row, locally and exactly.

        Offline and key-free. ``model`` selects the encoding (``gpt-4o`` /
        ``gpt-4.1`` / ``o``-series → ``o200k_base``; ``gpt-4`` / ``gpt-3.5`` →
        ``cl100k_base``); unknown names fall back to ``o200k_base``. Returns
        ``Int64`` — null in → null out, ``""`` → 0.

        Uses the native Rust accelerator (``polars-llm-accel``) as an in-engine
        Polars expression when installed; otherwise a ``tiktoken`` UDF.
        """
        return openai_tokens_expr(self._prompt, model=model, with_metadata=with_metadata, on_error=on_error)

    def gemini_tokens(
        self,
        *,
        model: str | None = None,
        exact: bool = False,
        client: Any = None,
        tokenizer_path: str | None = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Count Gemini tokens per row.

        Default (``exact=False``): local, exact, key-free — Gemini shares the
        public **Gemma** SentencePiece tokenizer, whose counts match Google's API
        with no network. Uses the native Rust accelerator (``polars-llm-accel``)
        when a local ``tokenizer_path`` (or ``POLARS_LLM_GEMMA_TOKENIZER``) is set,
        else an HF ``tokenizers`` UDF. ``exact=True`` counts via the Gemini API
        (e.g. multimodal inputs or cross-checking), batched like the chat verbs.
        """
        if not exact:
            return gemini_offline_expr(
                self._prompt,
                tokenizer_path=tokenizer_path,
                with_metadata=with_metadata,
                on_error=on_error,
            )

        chat = _make_chat("gemini", model, client, model_kwargs)

        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return count_batch_sync(chat.get_num_tokens, texts, retries=retries, backoff=backoff, cache=cache)

        return tokens_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    def agemini_tokens(
        self,
        *,
        model: str | None = None,
        exact: bool = False,
        client: Any = None,
        tokenizer_path: str | None = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Async sibling of :meth:`gemini_tokens`.

        ``max_concurrency`` applies to the ``exact=True`` API path; with
        ``exact=False`` the local tokenizer is already used, so it delegates to
        the offline path.
        """
        if not exact:
            return self.gemini_tokens(
                model=model,
                tokenizer_path=tokenizer_path,
                with_metadata=with_metadata,
                on_error=on_error,
            )
        chat = _make_chat("gemini", model, client, model_kwargs)
        counter = chat.get_num_tokens

        async def acounter(text: str) -> int:
            return await asyncio.to_thread(counter, text)

        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return _arun(
                count_batch_async(
                    acounter,
                    texts,
                    retries=retries,
                    backoff=backoff,
                    max_concurrency=max_concurrency,
                    cache=cache,
                ),
            )

        return tokens_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    def _anthropic_offline(self, chars_per_token: float, with_metadata: bool) -> pl.Expr:
        n_expr = anthropic_offline_expr(self._prompt, chars_per_token)
        if not with_metadata:
            return n_expr
        return pl.struct(
            n_expr.alias("tokens"),
            pl.lit(None, dtype=pl.Float64).alias("elapsed_ms"),
            pl.lit(None, dtype=pl.Utf8).alias("error"),
        )

    def anthropic_tokens(
        self,
        *,
        model: str | None = None,
        exact: bool = False,
        chars_per_token: float = 3.5,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Count Anthropic (Claude) tokens per row.

        Claude has no public tokenizer, so the default (``exact=False``) is an
        *estimate* using Anthropic's documented heuristic of ~1 token per
        ``chars_per_token`` characters — offline, key-free, and lowered to native
        Polars arithmetic. ``exact=True`` calls the ``count_tokens`` API, batched
        like the chat verbs.
        """
        if not exact:
            return self._anthropic_offline(chars_per_token, with_metadata)
        chat = _make_chat("anthropic", model, client, model_kwargs)
        counter = lambda text: chat.get_num_tokens_from_messages([HumanMessage(content=text)])

        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return count_batch_sync(counter, texts, retries=retries, backoff=backoff, cache=cache)

        return tokens_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    def aanthropic_tokens(
        self,
        *,
        model: str | None = None,
        exact: bool = False,
        chars_per_token: float = 3.5,
        client: Any = None,
        retries: int = 0,
        backoff: float = 0.0,
        max_concurrency: int | None = None,
        cache: bool = False,
        with_metadata: bool = False,
        on_error: OnError = "null",
        **model_kwargs: Any,
    ) -> pl.Expr:
        """Async sibling of :meth:`anthropic_tokens`.

        ``max_concurrency`` applies to the ``exact=True`` API path; the offline
        heuristic has no network, so ``exact=False`` delegates to it.
        """
        if not exact:
            return self._anthropic_offline(chars_per_token, with_metadata)
        chat = _make_chat("anthropic", model, client, model_kwargs)
        sync_counter = lambda text: chat.get_num_tokens_from_messages([HumanMessage(content=text)])

        async def acounter(text: str) -> int:
            return await asyncio.to_thread(sync_counter, text)

        def runner(texts: list[Any]) -> list[dict[str, Any]]:
            return _arun(
                count_batch_async(
                    acounter,
                    texts,
                    retries=retries,
                    backoff=backoff,
                    max_concurrency=max_concurrency,
                    cache=cache,
                ),
            )

        return tokens_map_batches(self._prompt, runner, with_metadata=with_metadata, on_error=on_error)

    def cost(
        self,
        *,
        model: str,
        kind: Literal["input", "output"] = "input",
        provider: Literal["openai", "anthropic", "gemini"] | None = None,
        exact: bool = False,
        prices: dict[str, Price] | None = None,
        **token_kwargs: Any,
    ) -> pl.Expr:
        """Estimate USD cost per row as ``tokens * price``.

        Counts tokens with the matching provider verb (``provider`` inferred from
        ``model``, or forced explicitly) and multiplies by the per-token price
        from ``polars_llm.PRICES``. ``kind="output"`` prices a generated-text
        column at the output rate, so the same verb covers input and output.
        Override prices per call with ``prices={...}`` or globally by mutating
        ``polars_llm.PRICES``. Returns ``Float64``.
        """
        prov = provider or infer_provider(model)
        if prov == "openai":
            tokens = self.openai_tokens(model=model)
        elif prov == "anthropic":
            tokens = self.anthropic_tokens(model=model, exact=exact, **token_kwargs)
        else:  # gemini
            tokens = self.gemini_tokens(model=model, exact=exact, **token_kwargs)
        return tokens.cast(pl.Float64) * price_per_token(model, kind, prices)

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
        elif isinstance(other, (list, tuple)):
            b = pl.lit(pl.Series("", [list(other)], dtype=list_dtype))
        else:
            raise TypeError(
                f"polars-llm: `cosine` expects a pl.Expr, pl.Series, or list of floats; got {type(other).__name__}",
            )
        dot = (a * b).list.sum()
        norm_a = (a * a).list.sum().sqrt()
        norm_b = (b * b).list.sum().sqrt()
        return dot / (norm_a * norm_b)
