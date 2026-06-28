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

## Token counting & cost verbs

Require the `tokenizers` extra (`pip install "polars-llm[tokenizers]"`).

| Method                                                   | Provider      | Offline default     |
| -------------------------------------------------------- | ------------- | ------------------- |
| [`openai_tokens`](#polars_llm.llm.Llm.openai_tokens)     | OpenAI        | exact (tiktoken)    |
| [`gemini_tokens`](#polars_llm.llm.Llm.gemini_tokens)     | Google Gemini | exact (Gemma)       |
| [`agemini_tokens`](#polars_llm.llm.Llm.agemini_tokens)   | Google Gemini | exact (Gemma)       |
| [`anthropic_tokens`](#polars_llm.llm.Llm.anthropic_tokens) | Anthropic   | estimate (heuristic) |
| [`aanthropic_tokens`](#polars_llm.llm.Llm.aanthropic_tokens) | Anthropic | estimate (heuristic) |
| [`cost`](#polars_llm.llm.Llm.cost)                       | any           | tokens × price      |

Counting verbs return an `Int64` column (or a `Struct{tokens, elapsed_ms, error}` with `with_metadata=True`); `cost` returns `Float64`. OpenAI and Gemini count exactly and offline; Anthropic counts offline via its published ~3.5-chars/token heuristic, or via the `count_tokens` API with `exact=True`. Prices come from `polars_llm.PRICES` (overridable).

## `polars_llm.Llm`

::: polars_llm.llm.Llm
