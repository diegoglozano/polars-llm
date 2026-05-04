---
title: API reference
description: Full API reference for polars-llm — the .ai expression namespace registered on Polars.
---

# API reference

`polars-llm` registers an `ai` namespace on every Polars expression. Import the package once and the namespace becomes available on any expression that resolves to a string column (the prompt).

```python
import polars as pl
import polars_llm  # noqa: F401  — registers the `.ai` namespace
```

## Chat verbs

| Method                                   | Provider      | Mode  |
| ---------------------------------------- | ------------- | ----- |
| [`openai`](#polars_llm.ai.Ai.openai)     | OpenAI        | sync  |
| [`aopenai`](#polars_llm.ai.Ai.aopenai)   | OpenAI        | async |
| [`anthropic`](#polars_llm.ai.Ai.anthropic) | Anthropic   | sync  |
| [`aanthropic`](#polars_llm.ai.Ai.aanthropic) | Anthropic | async |
| [`gemini`](#polars_llm.ai.Ai.gemini)     | Google Gemini | sync  |
| [`agemini`](#polars_llm.ai.Ai.agemini)   | Google Gemini | async |

Chat verbs return a `Utf8` column with the model's response. With `schema=`, they return a struct column matching the Pydantic model.

## Embedding verbs

| Method                                              | Provider          | Mode  |
| --------------------------------------------------- | ----------------- | ----- |
| [`openai_embed`](#polars_llm.ai.Ai.openai_embed)    | OpenAI Embeddings | sync  |
| [`aopenai_embed`](#polars_llm.ai.Ai.aopenai_embed)  | OpenAI Embeddings | async |
| [`gemini_embed`](#polars_llm.ai.Ai.gemini_embed)    | Google Gemini     | sync  |
| [`agemini_embed`](#polars_llm.ai.Ai.agemini_embed)  | Google Gemini     | async |

Embedding verbs return a `List[Float64]` column.

## `polars_llm.Ai`

::: polars_llm.ai.Ai
