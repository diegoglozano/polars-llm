"""Tests for the polars_llm `.llm` namespace.

The provider classes (`ChatOpenAI`, `ChatAnthropic`, …, `OpenAIEmbeddings`,
`GoogleGenerativeAIEmbeddings`) are monkey-patched with LangChain-native test
fakes — subclasses of `BaseChatModel` / `Embeddings` — so we exercise the full
LangChain Runnable contract (`invoke`, `ainvoke`, `batch`, `with_structured_output`,
…) without needing API keys at test time.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any, Callable

import polars as pl
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, PrivateAttr

import polars_llm  # noqa: F401  registers the `.llm` namespace
from polars_llm import llm as _llm_module


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class CallbackChat(BaseChatModel):
    """A `BaseChatModel` test fake that delegates each call to a Python callback.

    Built on `BaseChatModel`, so the full Runnable contract is exercised:
    `invoke`, `ainvoke`, `batch`, `abatch`, `with_structured_output`, and
    callback machinery. `responder(messages) -> str | AIMessage | BaseException`
    drives the response per call. `fail_first=N` raises on the first N calls
    (for retry tests). `delay` makes async calls await before responding (for
    concurrency-cap tests).
    """

    _responder: Callable[[list[BaseMessage]], Any] = PrivateAttr()
    _fail_first: int = PrivateAttr(default=0)
    _delay: float = PrivateAttr(default=0.0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _in_flight: int = PrivateAttr(default=0)
    _in_flight_max: int = PrivateAttr(default=0)

    def __init__(
        self,
        responder: Callable[[list[BaseMessage]], Any] | None = None,
        *,
        fail_first: int = 0,
        delay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responder = responder or (lambda msgs: "ok")
        self._fail_first = fail_first
        self._delay = delay

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def in_flight_max(self) -> int:
        return self._in_flight_max

    @property
    def _llm_type(self) -> str:
        return "callback-fake"

    def _make_message(self, messages: list[BaseMessage]) -> AIMessage:
        self._calls.append(messages)
        if len(self._calls) <= self._fail_first:
            raise RuntimeError(f"simulated failure #{len(self._calls)}")
        result = self._responder(messages)
        if isinstance(result, BaseException):
            raise result
        return AIMessage(content=result if isinstance(result, str) else str(result))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._make_message(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._in_flight += 1
        self._in_flight_max = max(self._in_flight_max, self._in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return self._generate(messages, stop, run_manager, **kwargs)
        finally:
            self._in_flight -= 1

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Pipe the AIMessage's content through PydanticOutputParser so the
        # downstream code receives a real Pydantic instance — same shape as
        # production providers' `with_structured_output`.
        return self | PydanticOutputParser(pydantic_object=schema)


class CallbackEmbeddings(Embeddings):
    """An `Embeddings` test fake driven by a Python callback."""

    def __init__(
        self,
        fn: Callable[[str], list[float]] | None = None,
        *,
        fail_first: int = 0,
    ) -> None:
        self._fn = fn or (lambda t: [float(len(t)), 0.0, 0.0])
        self._fail_first = fail_first
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if len(self.calls) <= self._fail_first:
            raise RuntimeError("simulated embed failure")
        return self._fn(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.aembed_query(t) for t in texts]


def _patch_chat(monkeypatch: pytest.MonkeyPatch, attr: str, fake: Any) -> None:
    monkeypatch.setattr(f"polars_llm.llm.{attr}", lambda **_: fake)


def _patch_embed(monkeypatch: pytest.MonkeyPatch, attr: str, fake: Any) -> None:
    monkeypatch.setattr(f"polars_llm.llm.{attr}", lambda **_: fake)


# --------------------------------------------------------------------------
# Chat: basic verbs
# --------------------------------------------------------------------------
def test_openai_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: f"echo:{msgs[-1].content}")
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["hello", "world"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="gpt-4o-mini").alias("res"))

    assert out["res"].to_list() == ["echo:hello", "echo:world"]


def test_anthropic_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: f"a:{msgs[-1].content}")
    _patch_chat(monkeypatch, "ChatAnthropic", fake)

    df = pl.DataFrame({"prompt": ["x", "y"]})
    out = df.with_columns(pl.col("prompt").llm.anthropic(model="claude-sonnet-4-6").alias("r"))

    assert out["r"].to_list() == ["a:x", "a:y"]


def test_gemini_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: f"g:{msgs[-1].content}")
    _patch_chat(monkeypatch, "ChatGoogleGenerativeAI", fake)

    df = pl.DataFrame({"prompt": ["q"]})
    out = df.with_columns(pl.col("prompt").llm.gemini(model="gemini-2.5-pro").alias("r"))

    assert out["r"].to_list() == ["g:q"]


# --------------------------------------------------------------------------
# Chat: async verbs
# --------------------------------------------------------------------------
def test_aopenai_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: msgs[-1].content.upper())
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["a", "b", "c"]})
    out = df.with_columns(pl.col("prompt").llm.aopenai(model="gpt-4o-mini").alias("r"))

    assert out["r"].to_list() == ["A", "B", "C"]


def test_aanthropic_aopenai_agemini_run(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr, verb, model in [
        ("ChatOpenAI", "aopenai", "gpt-4o-mini"),
        ("ChatAnthropic", "aanthropic", "claude-sonnet-4-6"),
        ("ChatGoogleGenerativeAI", "agemini", "gemini-2.5-pro"),
    ]:
        fake = CallbackChat(lambda msgs: "ok")
        _patch_chat(monkeypatch, attr, fake)
        df = pl.DataFrame({"prompt": ["x"]})
        method = getattr(pl.col("prompt").llm, verb)
        out = df.with_columns(method(model=model).alias("r"))
        assert out["r"].to_list() == ["ok"]


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
def test_system_prompt_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def responder(msgs: list[Any]) -> str:
        seen.append([type(m).__name__ for m in msgs])
        return "ok"

    fake = CallbackChat(responder)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["hi"]})
    df.with_columns(pl.col("prompt").llm.openai(model="x", system="be terse").alias("r")).collect_schema()

    assert seen == [["SystemMessage", "HumanMessage"]]
    assert fake.calls[0][0].content == "be terse"
    assert fake.calls[0][1].content == "hi"


def test_system_prompt_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_systems: list[str] = []

    def responder(msgs: list[Any]) -> str:
        # SystemMessage is first when present
        if type(msgs[0]).__name__ == "SystemMessage":
            seen_systems.append(msgs[0].content)
        return "ok"

    fake = CallbackChat(responder)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["a", "b"], "sys": ["s1", "s2"]})
    df.with_columns(
        pl.col("prompt").llm.openai(model="x", system=pl.col("sys")).alias("r"),
    ).collect_schema()

    assert sorted(seen_systems) == ["s1", "s2"]


def test_no_system_message_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_types: list[list[str]] = []

    def responder(msgs: list[Any]) -> str:
        seen_types.append([type(m).__name__ for m in msgs])
        return "ok"

    fake = CallbackChat(responder)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["hi"]})
    df.with_columns(pl.col("prompt").llm.openai(model="x").alias("r")).collect_schema()

    assert seen_types == [["HumanMessage"]]


# --------------------------------------------------------------------------
# Metadata, errors, retries
# --------------------------------------------------------------------------
def test_with_metadata_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: "hello")
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="m", with_metadata=True).alias("r"))

    row = out["r"].to_list()[0]
    assert row["content"] == "hello"
    assert row["error"] is None
    assert row["elapsed_ms"] >= 0


def test_with_metadata_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: ValueError("bad"))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="m", with_metadata=True).alias("r"))

    row = out["r"].to_list()[0]
    assert row["content"] is None
    assert "ValueError" in row["error"]


def test_retries_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: "ok", fail_first=2)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="m", retries=3, backoff=0.0).alias("r"))

    assert out["r"].to_list() == ["ok"]
    assert len(fake.calls) == 3


def test_retries_exhausted_returns_null(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: RuntimeError("boom"))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    with pytest.warns(UserWarning, match=r"1/1 request\(s\) failed"):
        out = df.with_columns(pl.col("prompt").llm.openai(model="m", retries=2, backoff=0.0).alias("r"))

    assert out["r"].to_list() == [None]


def test_on_error_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: RuntimeError("boom"))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    with pytest.raises(Exception, match="boom"):
        df.with_columns(pl.col("prompt").llm.openai(model="m", on_error="raise").alias("r"))


def test_no_warning_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: RuntimeError("boom"))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df.with_columns(pl.col("prompt").llm.openai(model="m", with_metadata=True).alias("r"))


# --------------------------------------------------------------------------
# Cache, concurrency
# --------------------------------------------------------------------------
def test_cache_dedupes_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: msgs[-1].content)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["a", "a", "b", "a"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="m", cache=True).alias("r"))

    assert out["r"].to_list() == ["a", "a", "b", "a"]
    assert len(fake.calls) == 2


def test_cache_dedupes_async(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: msgs[-1].content)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["a", "b", "a", "b"]})
    out = df.with_columns(pl.col("prompt").llm.aopenai(model="m", cache=True).alias("r"))

    assert out["r"].to_list() == ["a", "b", "a", "b"]
    assert len(fake.calls) == 2


def test_max_concurrency_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: "ok", delay=0.01)
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": [f"p{i}" for i in range(10)]})
    out = df.with_columns(pl.col("prompt").llm.aopenai(model="m", max_concurrency=2).alias("r"))

    assert out["r"].to_list() == ["ok"] * 10
    assert fake.in_flight_max <= 2


# --------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------
class _Person(BaseModel):
    name: str
    age: int


def test_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: json.dumps({"name": "Diego", "age": 30}))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["who?"]})
    out = df.with_columns(pl.col("prompt").llm.openai(model="m", schema=_Person).alias("r"))

    row = out["r"].to_list()[0]
    assert row == {"name": "Diego", "age": 30}


def test_structured_output_async(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackChat(lambda msgs: json.dumps({"name": "Ana", "age": 25}))
    _patch_chat(monkeypatch, "ChatOpenAI", fake)

    df = pl.DataFrame({"prompt": ["who?"]})
    out = df.with_columns(pl.col("prompt").llm.aopenai(model="m", schema=_Person).alias("r"))

    row = out["r"].to_list()[0]
    assert row == {"name": "Ana", "age": 25}


# --------------------------------------------------------------------------
# Custom client passthrough
# --------------------------------------------------------------------------
def test_custom_client_is_used() -> None:
    fake = CallbackChat(lambda msgs: "from-client")

    df = pl.DataFrame({"prompt": ["x", "y"]})
    out = df.with_columns(pl.col("prompt").llm.openai(client=fake).alias("r"))

    assert out["r"].to_list() == ["from-client", "from-client"]
    assert len(fake.calls) == 2


def test_model_required_when_no_client() -> None:
    with pytest.raises(ValueError, match="model"):
        pl.DataFrame({"prompt": ["x"]}).with_columns(pl.col("prompt").llm.openai().alias("r"))


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def test_openai_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [float(len(t)), 1.0, 2.0])
    _patch_embed(monkeypatch, "OpenAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["hi", "hello"]})
    out = df.with_columns(pl.col("prompt").llm.openai_embed(model="text-embedding-3-small").alias("v"))

    vectors = out["v"].to_list()
    assert vectors == [[2.0, 1.0, 2.0], [5.0, 1.0, 2.0]]


def test_aopenai_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [1.0, 2.0])
    _patch_embed(monkeypatch, "OpenAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["a", "b"]})
    out = df.with_columns(pl.col("prompt").llm.aopenai_embed(model="m").alias("v"))

    assert out["v"].to_list() == [[1.0, 2.0], [1.0, 2.0]]


def test_gemini_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [0.5])
    _patch_embed(monkeypatch, "GoogleGenerativeAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    out = df.with_columns(pl.col("prompt").llm.gemini_embed(model="m").alias("v"))

    assert out["v"].to_list() == [[0.5]]


def test_embed_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [0.1, 0.2, 0.3])
    _patch_embed(monkeypatch, "OpenAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    out = df.with_columns(pl.col("prompt").llm.openai_embed(model="m", with_metadata=True).alias("v"))

    row = out["v"].to_list()[0]
    assert row["vector"] == [0.1, 0.2, 0.3]
    assert row["dim"] == 3
    assert row["error"] is None


def test_embed_failure_returns_null_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [0.0], fail_first=10)
    _patch_embed(monkeypatch, "OpenAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["x"]})
    with pytest.warns(UserWarning, match=r"1/1 request\(s\) failed"):
        out = df.with_columns(pl.col("prompt").llm.openai_embed(model="m").alias("v"))

    assert out["v"].to_list() == [None]


def test_embed_cache_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = CallbackEmbeddings(lambda t: [float(len(t))])
    _patch_embed(monkeypatch, "OpenAIEmbeddings", fake)

    df = pl.DataFrame({"prompt": ["a", "a", "bb", "a"]})
    out = df.with_columns(pl.col("prompt").llm.openai_embed(model="m", cache=True).alias("v"))

    assert out["v"].to_list() == [[1.0], [1.0], [2.0], [1.0]]
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------
# Optional-extra import errors
# --------------------------------------------------------------------------
def test_missing_provider_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_llm_module, "ChatOpenAI", None)

    df = pl.DataFrame({"prompt": ["x"]})
    with pytest.raises(ImportError, match=r"polars-llm\[openai\]"):
        df.with_columns(pl.col("prompt").llm.openai(model="m").alias("r"))
