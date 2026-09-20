---
title: polars-llm — LLM and embedding calls from Polars DataFrames
description: Call chat, TypeSafe decision, and embedding models from a Polars DataFrame using native Polars expressions.
---

# polars-llm

[![PyPI version](https://img.shields.io/pypi/v/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Build status](https://img.shields.io/github/actions/workflow/status/diegoglozano/polars-llm/main.yml?branch=main)](https://github.com/diegoglozano/polars-llm/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/diegoglozano/polars-llm)](https://github.com/diegoglozano/polars-llm/blob/main/LICENSE)

**Call chat, TypeSafe decision, and embedding models from a [Polars](https://pola.rs) DataFrame, one row at a time, using native Polars expressions.**

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
- **Typed decisions** — run TypeSafe `Choice`, `Score`, and `Noul` questions together and get probabilities and confidence as nested structs.
- **Embeddings** — `openai_embed` and `gemini_embed` return `List[Float64]` columns.
- **Vector search** — compare vectors in an expression or run a top-K nearest-neighbor join between DataFrames.
- **Powered by [LangChain](https://python.langchain.com/)**.

## Install

Python 3.10 or newer is required. Python 3.9 is no longer supported.

```sh
pip install "polars-llm[openai]"
```

Choose a different extra for Anthropic, Gemini, TypeSafe, or nearest-neighbor search. See [Getting started](getting-started.md#install-a-provider) for all installation options and environment variables.

## Quickstart

### Chat per row

```python
import polars as pl
import polars_llm  # noqa: F401

df = (
    pl.DataFrame({"user_prompt": ["Capital of Spain?", "Capital of France?"]})
      .with_columns(
          pl.col("user_prompt").llm.openai(model="gpt-4o-mini").alias("answer")
      )
)
```

The result is an ordinary DataFrame with a new `answer` column. From there it can be filtered, joined, grouped, or written with the rest of your Polars pipeline.

## Where next?

- [Getting started](getting-started.md) covers installation, authentication, and your first end-to-end pipeline.
- [Examples](examples.md) has recipes for prompts built from columns, structured extraction, concurrent calls, embeddings, and vector search.
- [API reference](documentation.md) lists every expression and DataFrame method.

## Project links

- **GitHub**: <https://github.com/diegoglozano/polars-llm>
- **PyPI**: <https://pypi.org/project/polars-llm/>
- **Issues**: <https://github.com/diegoglozano/polars-llm/issues>
