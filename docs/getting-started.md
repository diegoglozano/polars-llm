---
title: Getting started
description: Install polars-llm, configure a provider, and run an LLM over a Polars DataFrame.
---

# Getting started

This guide takes a small DataFrame, calls a chat model once for each row, and adds the responses as a new column.

## Requirements

- Python 3.10 or newer. Python 3.9 is no longer supported.
- Polars 1.0 or newer.
- Credentials for at least one model provider.

Check your interpreter before installing:

```console
$ python --version
Python 3.12.7
```

If this reports Python 3.9, create a new environment with Python 3.10 or later before continuing.

## Install a provider

The base package is deliberately small. Install the extra for the provider or feature you use:

=== "OpenAI"

    ```sh
    pip install "polars-llm[openai]"
    ```

=== "Anthropic"

    ```sh
    pip install "polars-llm[anthropic]"
    ```

=== "Gemini"

    ```sh
    pip install "polars-llm[gemini]"
    ```

=== "TypeSafe"

    ```sh
    pip install "polars-llm[typesafe]"
    ```

=== "Everything"

    ```sh
    pip install "polars-llm[all]"
    ```

With `uv`, replace `pip install` with `uv add`. The optional `[ann]` extra installs NumPy and usearch for nearest-neighbor joins.

## Configure credentials

Set the environment variable expected by your provider:

| Provider | Environment variable |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| TypeSafe | `TYPESAFE_API_KEY` |

For example:

```sh
export OPENAI_API_KEY="your-api-key"
```

Keep API keys out of source code and version control.

## Run your first pipeline

Importing `polars_llm` registers the `.llm` namespace on Polars expressions. The import must remain even when your editor marks it as unused.

```python
import polars as pl
import polars_llm  # noqa: F401

tickets = pl.DataFrame(
    {
        "ticket_id": [101, 102, 103],
        "message": [
            "My order arrived damaged.",
            "Where can I download my invoice?",
            "Please change the delivery address.",
        ],
    }
)

result = tickets.with_columns(
    pl.col("message")
    .llm.openai(
        model="gpt-4o-mini",
        system="Classify the request as returns, billing, or shipping. Reply with only the label.",
    )
    .alias("category")
)

print(result)
```

Each value in `message` becomes a user message. The returned strings become the `category` column, while `ticket_id` and `message` remain unchanged.

!!! note "Provider calls happen when Polars evaluates the expression"

    In a lazy pipeline, no request is sent until `collect()` runs. Repeatedly collecting the same lazy query sends the requests again unless an external cache handles them.

## Switch providers

The provider methods share the same shape, so the surrounding Polars pipeline does not change:

```python
# Anthropic
pl.col("message").llm.anthropic(
    model="claude-sonnet-4-6",
    system="Return one category.",
)

# Gemini
pl.col("message").llm.gemini(
    model="gemini-2.5-pro",
    system="Return one category.",
)
```

Use the matching installation extra and API key for the provider you select.

## Next steps

- Use a Pydantic model for typed results in [Structured extraction](examples.md#structured-extraction).
- Improve throughput with [Concurrent calls](examples.md#concurrent-calls).
- Inspect failures without losing the whole batch in [Retries and error metadata](examples.md#retries-and-error-metadata).
- Build semantic search in [Embeddings and nearest-neighbor search](examples.md#embeddings-and-nearest-neighbor-search).
