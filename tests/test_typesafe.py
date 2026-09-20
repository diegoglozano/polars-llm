"""Tests for TypeSafe System One expressions."""

from __future__ import annotations

import asyncio
from typing import Any

import polars as pl
import pytest

import polars_llm  # noqa: F401
from polars_llm import llm as _llm_module

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"returns": "Exchanges", "billing": "Charges"},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is this?",
        "criteria": ["Cosmetic", "Degraded", "Blocking"],
    },
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}


def _response(state: Any, model: str | None) -> dict[str, Any]:
    text = str(state)
    returns = "size" in text
    return {
        "model": model or "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "returns" if returns else "billing",
                "probabilities": {
                    "returns": 0.9 if returns else 0.1,
                    "billing": 0.1 if returns else 0.9,
                },
                "confidence": 0.8,
            },
            "severity": {
                "type": "score",
                "score": 1.25,
                "probabilities": {0: 0.0, 1: 0.75, 2: 0.25},
                "legend": {0: "Cosmetic", 1: "Degraded", 2: "Blocking"},
                "confidence": 0.5,
            },
            "urgent": {"type": "noul", "noul": 0.72},
        },
        "usage": {"input_tokens": 20, "output_tokens": 8},
    }


class FakeTypeSafeClient:
    def __init__(self, *, fail_first: int = 0) -> None:
        self.calls: list[Any] = []
        self.fail_first = fail_first

    def system_one(self, *, state: Any, questions: Any, model: str | None) -> dict[str, Any]:
        self.calls.append(state)
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("temporary failure")
        return _response(state, model)


class FakeAsyncTypeSafeClient:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls: list[Any] = []
        self.delay = delay
        self.in_flight = 0
        self.in_flight_max = 0

    async def system_one(self, *, state: Any, questions: Any, model: str | None) -> dict[str, Any]:
        self.calls.append(state)
        self.in_flight += 1
        self.in_flight_max = max(self.in_flight_max, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return _response(state, model)
        finally:
            self.in_flight -= 1


def test_typesafe_returns_typed_nested_struct() -> None:
    client = FakeTypeSafeClient()
    frame = pl.DataFrame({"ticket": ["wrong size", "charged twice"]})

    output = frame.with_columns(
        pl.col("ticket").llm.typesafe(questions=QUESTIONS, client=client).alias("decision"),
    )

    rows = output["decision"].to_list()
    assert rows[0]["department"]["choice"] == "returns"
    assert rows[1]["department"]["probabilities"] == {"returns": 0.1, "billing": 0.9}
    assert rows[0]["severity"]["score"] == 1.25
    assert rows[0]["severity"]["probabilities"] == {"0": 0.0, "1": 0.75, "2": 0.25}
    assert rows[0]["urgent"]["noul"] == 0.72


def test_typesafe_accepts_struct_state_and_metadata() -> None:
    client = FakeTypeSafeClient()
    frame = pl.DataFrame({"ticket": ["charged twice"], "tier": ["business"]})

    output = frame.with_columns(
        pl
        .struct("ticket", "tier")
        .llm.typesafe(questions=QUESTIONS, model="jev-test", client=client, with_metadata=True)
        .alias("result"),
    )

    row = output["result"].to_list()[0]
    assert client.calls == [{"ticket": "charged twice", "tier": "business"}]
    assert row["model"] == "jev-test"
    assert row["input_tokens"] == 20
    assert row["output_tokens"] == 8
    assert row["answers"]["department"]["choice"] == "billing"
    assert row["error"] is None
    assert row["elapsed_ms"] >= 0


def test_typesafe_retries_and_caches_duplicate_states() -> None:
    client = FakeTypeSafeClient(fail_first=1)
    frame = pl.DataFrame({"ticket": ["wrong size", "wrong size"]})

    output = frame.with_columns(
        pl.col("ticket").llm.typesafe(questions=QUESTIONS, client=client, retries=1, cache=True).alias("decision"),
    )

    assert len(client.calls) == 2
    assert output["decision"].to_list()[0] == output["decision"].to_list()[1]


def test_atypesafe_runs_concurrently_with_cap() -> None:
    client = FakeAsyncTypeSafeClient(delay=0.01)
    frame = pl.DataFrame({"ticket": [str(index) for index in range(6)]})

    output = frame.with_columns(
        pl.col("ticket").llm.atypesafe(questions=QUESTIONS, client=client, max_concurrency=2).alias("decision"),
    )

    assert output.height == 6
    assert client.in_flight_max == 2


def test_typesafe_error_modes() -> None:
    frame = pl.DataFrame({"ticket": ["x"]})
    client = FakeTypeSafeClient(fail_first=1)
    with pytest.warns(UserWarning, match=r"1/1 request\(s\) failed"):
        output = frame.with_columns(
            pl.col("ticket").llm.typesafe(questions=QUESTIONS, client=client).alias("decision"),
        )
    assert output["decision"].to_list() == [None]

    with pytest.raises(Exception, match="temporary failure"):
        frame.with_columns(
            pl
            .col("ticket")
            .llm.typesafe(questions=QUESTIONS, client=FakeTypeSafeClient(fail_first=1), on_error="raise")
            .alias("decision"),
        )


def test_typesafe_validates_questions() -> None:
    expression = pl.col("ticket").llm
    with pytest.raises(ValueError, match="must not be empty"):
        expression.typesafe(questions={}, client=FakeTypeSafeClient())
    with pytest.raises(ValueError, match="at least two"):
        expression.typesafe(
            questions={"score": {"type": "score", "criteria": ["only one"]}},
            client=FakeTypeSafeClient(),
        )


def test_missing_typesafe_extra_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_llm_module, "TypeSafeClient", None)
    with pytest.raises(ImportError, match=r"polars-llm\[typesafe\]"):
        pl.col("ticket").llm.typesafe(questions=QUESTIONS)
