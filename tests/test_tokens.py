"""Tests for the polars_llm token-counting and cost verbs.

Offline paths (OpenAI tiktoken, the Anthropic char heuristic) are exact and
deterministic, so they get real golden-value assertions. The exact (API) paths
are exercised with fake counters passed via ``client=`` — ``_make_chat`` returns
the client verbatim when one is supplied, so no monkeypatching of provider
classes is needed. The local Gemini tokenizer is replaced with a fake (the real
Gemma model is a several-MB download).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import polars as pl
import pytest
import tiktoken

import polars_llm  # noqa: F401  registers the `.llm` namespace
from polars_llm import PRICES, Price, price_per_token
from polars_llm import _tokens as _tokens_module


def _tiktoken_data_available() -> bool:
    """tiktoken downloads its BPE vocab on first use; skip the OpenAI golden
    tests where that download is blocked (offline / restricted egress). CI has
    network, so the real assertions run there."""
    try:
        tiktoken.get_encoding("o200k_base")
        return True
    except Exception:
        return False


requires_tiktoken_data = pytest.mark.skipif(
    not _tiktoken_data_available(),
    reason="tiktoken BPE vocab download unavailable in this environment",
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeCounter:
    """Stands in for a LangChain chat model's token-count methods (passed as ``client=``).

    Counts words by default. ``fail_first=N`` raises on the first N calls (retry
    tests); ``delay`` sleeps before returning (concurrency tests). Call counts
    and max in-flight are tracked thread-safely so the async path can be probed.
    """

    def __init__(self, fn: Any = None, *, fail_first: int = 0, delay: float = 0.0) -> None:
        self._fn = fn or (lambda text: len(text.split()))
        self._fail_first = fail_first
        self._delay = delay
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.in_flight = 0
        self.in_flight_max = 0

    def _count(self, text: str) -> int:
        with self._lock:
            self.in_flight += 1
            self.in_flight_max = max(self.in_flight_max, self.in_flight)
            self.calls.append(text)
            should_fail = len(self.calls) <= self._fail_first
        try:
            if self._delay:
                time.sleep(self._delay)
            if should_fail:
                raise RuntimeError("boom")
            return self._fn(text)
        finally:
            with self._lock:
                self.in_flight -= 1

    def get_num_tokens_from_messages(self, messages: list[Any]) -> int:  # anthropic
        return self._count(messages[-1].content)

    def get_num_tokens(self, text: str) -> int:  # gemini
        return self._count(text)


class _FakeEnc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeGemma:
    """Fake HF ``tokenizers`` tokenizer — token count == word count."""

    def encode_batch(self, inputs: list[str]) -> list[_FakeEnc]:
        return [_FakeEnc(list(range(len(t.split())))) for t in inputs]


# --------------------------------------------------------------------------
# OpenAI — local exact (tiktoken)
# --------------------------------------------------------------------------
@requires_tiktoken_data
def test_openai_tokens_matches_tiktoken() -> None:
    texts = ["hello world", "polars + llm", "a much longer sentence with several tokens"]
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    expected = [len(enc.encode(t)) for t in texts]

    df = pl.DataFrame({"text": texts})
    out = df.with_columns(pl.col("text").llm.openai_tokens(model="gpt-4o-mini").alias("n"))

    assert out["n"].to_list() == expected
    assert out["n"].dtype == pl.Int64


@requires_tiktoken_data
def test_openai_tokens_unknown_model_falls_back_to_o200k() -> None:
    texts = ["fallback please"]
    expected = [len(tiktoken.get_encoding("o200k_base").encode(t)) for t in texts]

    df = pl.DataFrame({"text": texts})
    out = df.with_columns(pl.col("text").llm.openai_tokens(model="totally-made-up-model").alias("n"))

    assert out["n"].to_list() == expected


@requires_tiktoken_data
def test_openai_tokens_null_and_empty() -> None:
    df = pl.DataFrame({"text": ["", None]}, schema={"text": pl.Utf8})
    out = df.with_columns(pl.col("text").llm.openai_tokens(model="gpt-4o").alias("n"))

    assert out["n"].to_list() == [0, None]
    assert out["n"].dtype == pl.Int64


@requires_tiktoken_data
def test_openai_tokens_special_token_string_does_not_raise() -> None:
    df = pl.DataFrame({"text": ["<|endoftext|> and more"]})
    out = df.with_columns(pl.col("text").llm.openai_tokens(model="gpt-4o").alias("n"))

    assert out["n"][0] > 0


@requires_tiktoken_data
def test_openai_tokens_with_metadata_struct() -> None:
    df = pl.DataFrame({"text": ["hello world"]})
    out = df.with_columns(pl.col("text").llm.openai_tokens(model="gpt-4o", with_metadata=True).alias("m"))

    field_names = out["m"].struct.fields
    assert field_names == ["tokens", "elapsed_ms", "error"]
    assert out["m"].struct.field("tokens")[0] == 2
    assert out["m"].struct.field("error")[0] is None


# --------------------------------------------------------------------------
# Anthropic — offline heuristic (native Polars)
# --------------------------------------------------------------------------
def test_anthropic_tokens_offline_heuristic() -> None:
    texts = ["hello world", "x" * 35, ""]
    df = pl.DataFrame({"text": texts})
    out = df.with_columns(pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6").alias("n"))

    expected = [math.ceil(len(t) / 3.5) for t in texts]
    assert out["n"].to_list() == expected
    assert out["n"].dtype == pl.Int64


def test_anthropic_tokens_offline_custom_factor_and_null() -> None:
    df = pl.DataFrame({"text": ["hello world", None]}, schema={"text": pl.Utf8})
    out = df.with_columns(
        pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6", chars_per_token=2.0).alias("n"),
    )

    assert out["n"].to_list() == [math.ceil(11 / 2.0), None]


def test_anthropic_tokens_offline_with_metadata() -> None:
    df = pl.DataFrame({"text": ["hello world"]})
    out = df.with_columns(pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6", with_metadata=True).alias("m"))

    assert out["m"].struct.fields == ["tokens", "elapsed_ms", "error"]
    assert out["m"].struct.field("tokens")[0] == math.ceil(11 / 3.5)


def test_anthropic_tokens_invalid_chars_per_token() -> None:
    df = pl.DataFrame({"text": ["x"]})
    with pytest.raises(ValueError, match="chars_per_token"):
        df.with_columns(pl.col("text").llm.anthropic_tokens(model="claude-sonnet-4-6", chars_per_token=0).alias("n"))


# --------------------------------------------------------------------------
# Gemini — local exact (fake Gemma tokenizer)
# --------------------------------------------------------------------------
def test_gemini_tokens_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polars_llm.llm._gemma_tokenizer", lambda _path=None: FakeGemma())

    texts = ["one two three", "single", ""]
    df = pl.DataFrame({"text": texts})
    out = df.with_columns(pl.col("text").llm.gemini_tokens(model="gemini-2.5-pro").alias("n"))

    assert out["n"].to_list() == [3, 1, 0]
    assert out["n"].dtype == pl.Int64


# --------------------------------------------------------------------------
# Exact (API) paths — fake counters via client=
# --------------------------------------------------------------------------
def test_anthropic_tokens_exact_counts() -> None:
    fake = FakeCounter()
    df = pl.DataFrame({"text": ["one two", "a b c d"]})
    out = df.with_columns(pl.col("text").llm.anthropic_tokens(exact=True, client=fake).alias("n"))

    assert out["n"].to_list() == [2, 4]


def test_gemini_tokens_exact_counts() -> None:
    fake = FakeCounter()
    df = pl.DataFrame({"text": ["one two three"]})
    out = df.with_columns(pl.col("text").llm.gemini_tokens(exact=True, client=fake).alias("n"))

    assert out["n"].to_list() == [3]


def test_exact_dedup_with_cache() -> None:
    fake = FakeCounter()
    df = pl.DataFrame({"text": ["dup", "dup", "other"]})
    out = df.with_columns(pl.col("text").llm.anthropic_tokens(exact=True, client=fake, cache=True).alias("n"))

    assert out["n"].to_list() == [1, 1, 1]
    assert sorted(fake.calls) == ["dup", "other"]  # "dup" counted once


def test_exact_retries_recover() -> None:
    fake = FakeCounter(fail_first=1)
    df = pl.DataFrame({"text": ["alpha beta"]})
    out = df.with_columns(
        pl.col("text").llm.anthropic_tokens(exact=True, client=fake, retries=2).alias("n"),
    )

    assert out["n"].to_list() == [2]


def test_exact_on_error_null_warns() -> None:
    fake = FakeCounter(fail_first=10)
    df = pl.DataFrame({"text": ["boom"]})
    with pytest.warns(UserWarning):
        out = df.with_columns(pl.col("text").llm.anthropic_tokens(exact=True, client=fake).alias("n"))

    assert out["n"].to_list() == [None]


def test_exact_on_error_raise() -> None:
    fake = FakeCounter(fail_first=10)
    df = pl.DataFrame({"text": ["boom"]})
    with pytest.raises(RuntimeError):
        df.with_columns(pl.col("text").llm.anthropic_tokens(exact=True, client=fake, on_error="raise").alias("n"))


def test_exact_with_metadata() -> None:
    fake = FakeCounter()
    df = pl.DataFrame({"text": ["one two"]})
    out = df.with_columns(pl.col("text").llm.anthropic_tokens(exact=True, client=fake, with_metadata=True).alias("m"))

    assert out["m"].struct.field("tokens")[0] == 2
    assert out["m"].struct.field("error")[0] is None


def test_aanthropic_tokens_exact_concurrency_cap() -> None:
    fake = FakeCounter(delay=0.02)
    texts = [f"text number {i}" for i in range(6)]
    df = pl.DataFrame({"text": texts})
    out = df.with_columns(
        pl.col("text").llm.aanthropic_tokens(exact=True, client=fake, max_concurrency=2).alias("n"),
    )

    assert out["n"].to_list() == [3, 3, 3, 3, 3, 3]
    assert fake.in_flight_max <= 2


def test_agemini_tokens_offline_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polars_llm.llm._gemma_tokenizer", lambda _path=None: FakeGemma())
    df = pl.DataFrame({"text": ["one two three"]})
    out = df.with_columns(pl.col("text").llm.agemini_tokens(model="gemini-2.5-pro").alias("n"))

    assert out["n"].to_list() == [3]


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------
@requires_tiktoken_data
def test_cost_input_and_output() -> None:
    prices = {"gpt-4o": Price(10.0, 30.0)}
    df = pl.DataFrame({"text": ["hello world"]})  # tiktoken: 2 tokens
    out = df.with_columns(
        pl.col("text").llm.cost(model="gpt-4o", kind="input", prices=prices).alias("in_usd"),
        pl.col("text").llm.cost(model="gpt-4o", kind="output", prices=prices).alias("out_usd"),
    )

    assert out["in_usd"][0] == pytest.approx(2 * 10.0 / 1_000_000)
    assert out["out_usd"][0] == pytest.approx(2 * 30.0 / 1_000_000)
    assert out["in_usd"].dtype == pl.Float64


def test_cost_provider_override_anthropic() -> None:
    prices = {"my-claude": Price(3.0, 15.0)}
    df = pl.DataFrame({"text": ["x" * 35]})  # heuristic: ceil(35/3.5) = 10 tokens
    out = df.with_columns(
        pl.col("text").llm.cost(model="my-claude", provider="anthropic", prices=prices).alias("usd"),
    )

    assert out["usd"][0] == pytest.approx(10 * 3.0 / 1_000_000)


def test_cost_unknown_model_raises() -> None:
    df = pl.DataFrame({"text": ["x"]})
    with pytest.raises(ValueError, match="could not infer a provider"):
        df.with_columns(pl.col("text").llm.cost(model="mystery-model").alias("usd"))


def test_cost_no_price_entry_raises() -> None:
    df = pl.DataFrame({"text": ["x"]})
    with pytest.raises(ValueError, match="no price entry"):
        df.with_columns(pl.col("text").llm.cost(model="gpt-4o", prices={"other": Price(1.0, 1.0)}).alias("usd"))


# --------------------------------------------------------------------------
# Pricing helpers
# --------------------------------------------------------------------------
def test_price_per_token_and_registry() -> None:
    assert "gpt-4o" in PRICES
    assert price_per_token("gpt-4o-mini", "input", {"gpt-4o-mini": Price(0.15, 0.60)}) == pytest.approx(0.15 / 1e6)
    assert price_per_token("gpt-4o-mini", "output", {"gpt-4o-mini": Price(0.15, 0.60)}) == pytest.approx(0.60 / 1e6)


def test_price_per_token_bad_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        price_per_token("gpt-4o", "sideways")


@requires_tiktoken_data
def test_cost_module_level_prices_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_tokens_module.PRICES, "gpt-4o", Price(99.0, 99.0))
    df = pl.DataFrame({"text": ["hello world"]})  # 2 tokens
    out = df.with_columns(pl.col("text").llm.cost(model="gpt-4o", kind="input").alias("usd"))

    assert out["usd"][0] == pytest.approx(2 * 99.0 / 1_000_000)


# --------------------------------------------------------------------------
# Optional extra missing
# --------------------------------------------------------------------------
def test_tiktoken_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_tokens_module, "_TIKTOKEN", None)
    _tokens_module._encoding_for.cache_clear()
    df = pl.DataFrame({"text": ["hi"]})
    with pytest.raises(ImportError, match=r"polars-llm\[tokens\]"):
        df.with_columns(pl.col("text").llm.openai_tokens(model="gpt-4o").alias("n"))
    _tokens_module._encoding_for.cache_clear()


def test_tokenizers_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_tokens_module, "_TOKENIZERS", None)
    _tokens_module._gemma_tokenizer.cache_clear()
    df = pl.DataFrame({"text": ["hi"]})
    with pytest.raises(ImportError, match=r"polars-llm\[tokens\]"):
        df.with_columns(pl.col("text").llm.gemini_tokens(model="gemini-2.5-pro").alias("n"))
    _tokens_module._gemma_tokenizer.cache_clear()
