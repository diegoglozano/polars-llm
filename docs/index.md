---
title: polars-llm — LLM and embedding calls from Polars DataFrames
description: Call OpenAI, Anthropic, and Gemini chat and embedding models from a Polars DataFrame, one row at a time, using native Polars expressions. Powered by LangChain.
---

<p align="center">
  <img src="assets/logo-wordmark.svg" alt="polars-llm" width="420">
</p>

# polars-llm

[![PyPI version](https://img.shields.io/pypi/v/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Build status](https://img.shields.io/github/actions/workflow/status/diegoglozano/polars-llm/main.yml?branch=main)](https://github.com/diegoglozano/polars-llm/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/diegoglozano/polars-llm)](https://github.com/diegoglozano/polars-llm/blob/main/LICENSE)

**Call OpenAI, Anthropic, and Gemini models from a [Polars](https://pola.rs) DataFrame, one row at a time, using native Polars expressions.**

`polars-llm` registers an `.llm` namespace on Polars expressions so you can call any [LangChain](https://python.langchain.com/)-supported chat model or embedding model on every row of a DataFrame — synchronously or asynchronously — and pipe the responses straight back into your data pipeline.

```python
import polars as pl
import polars_llm  # noqa: F401  — registers the `.llm` namespace

(
    pl.DataFrame({"user_prompt": ["Summarise polars in one sentence."]})
      .with_columns(
          pl.col("user_prompt").llm.openai(model="gpt-4o-mini").alias("answer")
      )
)
```

## Why polars-llm?

- **Expression-native** — works inside `with_columns`, `select`, and any other Polars expression context.
- **Sync and async** — `aopenai`, `aanthropic`, `agemini` fan out concurrently with `asyncio.gather`.
- **Per-row prompts and system messages** — every argument can be a Polars expression.
- **Structured outputs** — pass a Pydantic schema as `schema=` and get a struct column back.
- **Embeddings** — `openai_embed` and `gemini_embed` return `List[Float64]` columns.
- **Powered by [LangChain](https://python.langchain.com/)**.

## Install

```sh
pip install "polars-llm[openai]"
pip install "polars-llm[anthropic]"
pip install "polars-llm[gemini]"
pip install "polars-llm[all]"
```

Requires Python 3.9+ and Polars 1.0+. Auth follows LangChain conventions: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` before importing.

## Quickstart

### Chat per row

```python
df = (
    pl.DataFrame({"user_prompt": ["Capital of Spain?", "Capital of France?"]})
      .with_columns(
          pl.col("user_prompt").llm.openai(model="gpt-4o-mini").alias("answer")
      )
)
```

### Structured output

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

df.with_columns(
    pl.col("review").llm.openai(model="gpt-4o-mini", schema=Sentiment).alias("s")
).unnest("s")
```

### Embeddings

```python
df.with_columns(
    pl.col("text").llm.openai_embed(model="text-embedding-3-small").alias("vector")
)
```

See the full [API reference](documentation.md).

## Links

- **GitHub**: <https://github.com/diegoglozano/polars-llm>
- **PyPI**: <https://pypi.org/project/polars-llm/>
- **Issues**: <https://github.com/diegoglozano/polars-llm/issues>
