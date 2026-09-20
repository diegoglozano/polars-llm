"""TypeSafe System One execution and Polars result shaping.

This module deliberately does not import :mod:`typesafe_sdk`.  The SDK is an
optional dependency and clients are constructed in :mod:`polars_llm.llm` only
when a TypeSafe expression is used.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import polars as pl

from ._runtime import OnError, _hashable, _warn_silent_errors


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _question_type(question: Any) -> str:
    question_type = _value(question, "type")
    if question_type not in {"noul", "choice", "score"}:
        raise ValueError(
            "polars-llm: each TypeSafe question must have type 'noul', 'choice', or 'score'",
        )
    return question_type


def _criteria(question: Any) -> Any:
    return _value(question, "criteria")


def typesafe_answers_dtype(questions: Mapping[str, Any]) -> pl.Struct:
    """Build the stable nested Struct dtype implied by a question mapping."""
    if not questions:
        raise ValueError("polars-llm: `questions` must not be empty")

    fields: dict[str, Any] = {}
    for raw_name, question in questions.items():
        name = str(raw_name)
        question_type = _question_type(question)
        if question_type == "noul":
            fields[name] = pl.Struct({"type": pl.Utf8, "noul": pl.Float64})
            continue

        criteria = _criteria(question)
        if question_type == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(
                    f"polars-llm: Choice question {name!r} requires a non-empty criteria mapping",
                )
            probability_dtype = pl.Struct({str(option): pl.Float64 for option in criteria})
            fields[name] = pl.Struct(
                {
                    "type": pl.Utf8,
                    "choice": pl.Utf8,
                    "probabilities": probability_dtype,
                    "confidence": pl.Float64,
                },
            )
            continue

        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)) or len(criteria) < 2:
            raise ValueError(
                f"polars-llm: Score question {name!r} requires at least two ordered criteria",
            )
        level_fields = {str(index): pl.Float64 for index in range(len(criteria))}
        legend_fields = {str(index): pl.Utf8 for index in range(len(criteria))}
        fields[name] = pl.Struct(
            {
                "type": pl.Utf8,
                "score": pl.Float64,
                "probabilities": pl.Struct(level_fields),
                "legend": pl.Struct(legend_fields),
                "confidence": pl.Float64,
            },
        )
    return pl.Struct(fields)


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:  # pragma: no cover - Pydantic v1/custom compatibility
            return value.model_dump()
    return {
        key: getattr(value, key)
        for key in ("type", "noul", "choice", "score", "probabilities", "legend", "confidence")
        if hasattr(value, key)
    }


def _json_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _keyed_values(values: Any, keys: Sequence[str], converter: Callable[[Any], Any]) -> dict[str, Any]:
    values = values or {}
    if not isinstance(values, Mapping):
        return dict.fromkeys(keys)
    return {key: converter(values.get(key, values.get(int(key) if key.isdigit() else key))) for key in keys}


def _normalise_response(response: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    answers = _value(response, "answers", {}) or {}
    normalised: dict[str, Any] = {}
    for raw_name, question in questions.items():
        name = str(raw_name)
        answer = answers.get(raw_name, answers.get(name)) if isinstance(answers, Mapping) else None
        if answer is None:
            normalised[name] = None
            continue

        raw = _model_dump(answer)
        question_type = _question_type(question)
        if question_type == "noul":
            normalised[name] = {"type": "noul", "noul": raw.get("noul")}
        elif question_type == "choice":
            option_names = [str(option) for option in _criteria(question)]
            normalised[name] = {
                "type": "choice",
                "choice": raw.get("choice"),
                "probabilities": _keyed_values(raw.get("probabilities"), option_names, lambda value: value),
                "confidence": raw.get("confidence"),
            }
        else:
            levels = [str(index) for index in range(len(_criteria(question)))]
            normalised[name] = {
                "type": "score",
                "score": raw.get("score"),
                "probabilities": _keyed_values(raw.get("probabilities"), levels, lambda value: value),
                "legend": _keyed_values(raw.get("legend"), levels, _json_text),
                "confidence": raw.get("confidence"),
            }

    usage = _value(response, "usage", {}) or {}
    return {
        "answers": normalised,
        "model": _value(response, "model"),
        "input_tokens": _value(usage, "input_tokens"),
        "output_tokens": _value(usage, "output_tokens"),
    }


def _error_result(start: float, exc: Exception) -> dict[str, Any]:
    return {
        "answers": None,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "elapsed_ms": (time.monotonic() - start) * 1000,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _typesafe_one_sync(
    client: Any,
    state: Any,
    questions: Mapping[str, Any],
    model: str | None,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    attempt = 0
    start = time.monotonic()
    while True:
        try:
            response = client.system_one(state=state, questions=questions, model=model)
            result = _normalise_response(response, questions)
            result.update(elapsed_ms=(time.monotonic() - start) * 1000, error=None)
            return result
        except Exception as exc:
            if attempt < retries:
                wait = backoff * (2**attempt) if backoff > 0 else 0.0
                if wait:
                    time.sleep(wait)
                attempt += 1
                continue
            return _error_result(start, exc)


def typesafe_batch_sync(
    client: Any,
    states: list[Any],
    *,
    questions: Mapping[str, Any],
    model: str | None,
    retries: int,
    backoff: float,
    cache: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    memo: dict[Any, dict[str, Any]] = {}
    for state in states:
        key = _hashable(state)
        if cache and key in memo:
            results.append(memo[key])
            continue
        result = _typesafe_one_sync(client, state, questions, model, retries, backoff)
        if cache:
            memo[key] = result
        results.append(result)
    return results


async def _typesafe_one_async(
    client: Any,
    semaphore: asyncio.Semaphore | None,
    state: Any,
    questions: Mapping[str, Any],
    model: str | None,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    async def _go() -> dict[str, Any]:
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                response = await client.system_one(state=state, questions=questions, model=model)
                result = _normalise_response(response, questions)
                result.update(elapsed_ms=(time.monotonic() - start) * 1000, error=None)
                return result
            except Exception as exc:
                if attempt < retries:
                    wait = backoff * (2**attempt) if backoff > 0 else 0.0
                    if wait:
                        await asyncio.sleep(wait)
                    attempt += 1
                    continue
                return _error_result(start, exc)

    if semaphore is None:
        return await _go()
    async with semaphore:
        return await _go()


async def typesafe_batch_async(
    client: Any,
    states: list[Any],
    *,
    questions: Mapping[str, Any],
    model: str | None,
    retries: int,
    backoff: float,
    max_concurrency: int | None,
    cache: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    unique_indices: dict[Any, int] = {}
    order: list[int] = []
    result_index: list[int] = [0] * len(states)
    for index, state in enumerate(states):
        key = _hashable(state)
        if cache and key in unique_indices:
            result_index[index] = unique_indices[key]
            continue
        result_index[index] = len(order)
        if cache:
            unique_indices[key] = len(order)
        order.append(index)

    tasks = [
        _typesafe_one_async(client, semaphore, states[index], questions, model, retries, backoff) for index in order
    ]
    unique_results = await asyncio.gather(*tasks)
    return [unique_results[result_index[index]] for index in range(len(states))]


def typesafe_results_to_series(
    results: list[dict[str, Any]],
    *,
    answers_dtype: pl.Struct,
    with_metadata: bool,
    on_error: OnError,
) -> pl.Series:
    if with_metadata:
        metadata_dtype = pl.Struct(
            {
                "answers": answers_dtype,
                "model": pl.Utf8,
                "input_tokens": pl.Int64,
                "output_tokens": pl.Int64,
                "elapsed_ms": pl.Float64,
                "error": pl.Utf8,
            },
        )
        return pl.Series(results, dtype=metadata_dtype)

    output: list[dict[str, Any] | None] = []
    silent_errors: list[str] = []
    for result in results:
        if result["error"] is None:
            output.append(result["answers"])
        elif on_error == "raise":
            raise RuntimeError(result["error"])
        else:
            output.append(None)
            silent_errors.append(result["error"])
    _warn_silent_errors(silent_errors, len(results))
    return pl.Series(output, dtype=answers_dtype)


def typesafe_map_batches(
    state_expr: pl.Expr,
    runner: Callable[[list[Any]], list[dict[str, Any]]],
    *,
    questions: Mapping[str, Any],
    with_metadata: bool,
    on_error: OnError,
) -> pl.Expr:
    answers_dtype = typesafe_answers_dtype(questions)
    if with_metadata:
        return_dtype: Any = pl.Struct(
            {
                "answers": answers_dtype,
                "model": pl.Utf8,
                "input_tokens": pl.Int64,
                "output_tokens": pl.Int64,
                "elapsed_ms": pl.Float64,
                "error": pl.Utf8,
            },
        )
    else:
        return_dtype = answers_dtype

    def _batch(series: pl.Series) -> pl.Series:
        return typesafe_results_to_series(
            runner(series.to_list()),
            answers_dtype=answers_dtype,
            with_metadata=with_metadata,
            on_error=on_error,
        )

    return state_expr.map_batches(_batch, return_dtype=return_dtype)
