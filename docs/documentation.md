---
title: API reference
description: Full API reference for polars-llm — the .llm expression namespace registered on Polars.
---

# API reference

`polars-llm` registers an `llm` namespace on every Polars expression. Import the package once and the namespace becomes available on any expression that resolves to a string column (the prompt).

```python
import polars as pl
import polars_llm  # noqa: F401  — registers the `.llm` namespace
```

## Chat verbs

| Method                                         | Provider      | Mode  |
| ---------------------------------------------- | ------------- | ----- |
| [`openai`](#polars_llm.llm.Llm.openai)         | OpenAI        | sync  |
| [`aopenai`](#polars_llm.llm.Llm.aopenai)       | OpenAI        | async |
| [`anthropic`](#polars_llm.llm.Llm.anthropic)   | Anthropic     | sync  |
| [`aanthropic`](#polars_llm.llm.Llm.aanthropic) | Anthropic     | async |
| [`gemini`](#polars_llm.llm.Llm.gemini)         | Google Gemini | sync  |
| [`agemini`](#polars_llm.llm.Llm.agemini)       | Google Gemini | async |

Chat verbs return a `Utf8` column with the model's response. With `schema=`, they return a struct column matching the Pydantic model.

## Embedding verbs

| Method                                               | Provider          | Mode  |
| ---------------------------------------------------- | ----------------- | ----- |
| [`openai_embed`](#polars_llm.llm.Llm.openai_embed)   | OpenAI Embeddings | sync  |
| [`aopenai_embed`](#polars_llm.llm.Llm.aopenai_embed) | OpenAI Embeddings | async |
| [`gemini_embed`](#polars_llm.llm.Llm.gemini_embed)   | Google Gemini     | sync  |
| [`agemini_embed`](#polars_llm.llm.Llm.agemini_embed) | Google Gemini     | async |

Embedding verbs return a `List[Float64]` column.

## TypeSafe decision verbs

| Method                                       | Provider            | Mode  |
| -------------------------------------------- | ------------------- | ----- |
| [`typesafe`](#polars_llm.llm.Llm.typesafe)   | TypeSafe System One | sync  |
| [`atypesafe`](#polars_llm.llm.Llm.atypesafe) | TypeSafe System One | async |

The source expression is evaluated as TypeSafe state. Pass `questions=` with any mix of `Choice`, `Score`, and `Noul` questions. These methods return a nested struct with one answer field per question; `with_metadata=True` also includes the model, token usage, elapsed time, and error. Install them with `pip install "polars-llm[typesafe]"` on Python 3.10 or newer.

## Vector helpers

| Method | Description |
| --- | --- |
| [`cosine`](#polars_llm.llm.Llm.cosine) | Compute cosine similarity with another vector expression or a literal vector. No provider call is made. |

## Nearest-neighbor joins

`polars_llm` also registers an `.ann` namespace on DataFrames. Use `df.ann.knn(other, ...)` to return the closest rows from another DataFrame. Both vector columns must have matching dimensions and use `List[Float32/64]` or `Array[Float32/64, dim]` values.

::: polars_llm._ann.Ann

## `polars_llm.Llm`

::: polars_llm.llm.Llm
