---
title: Examples
description: Practical polars-llm recipes for chat, structured output, concurrency, embeddings, and vector search.
---

# Examples

All examples assume `polars_llm` is installed with the relevant provider extra and its API key is set. Importing the package registers the `.llm` expression namespace and `.ann` DataFrame namespace.

```python
import polars as pl
import polars_llm  # noqa: F401
```

## Build prompts from columns

Use normal Polars expressions to prepare a prompt. `pl.format` keeps the prompt construction inside the query:

```python
reviews = pl.DataFrame(
    {
        "language": ["French", "Spanish"],
        "review": ["The delivery was quick.", "The package was damaged."],
    }
)

translated = reviews.with_columns(
    pl.format("Translate into {}: {}", "language", "review")
    .llm.openai(
        model="gpt-4o-mini",
        system="Translate accurately. Return only the translation.",
    )
    .alias("translation")
)
```

The system message can also vary by row. Pass an expression instead of a literal string:

```python
requests = pl.DataFrame(
    {
        "prompt": ["Explain vector search", "Explain a DataFrame"],
        "audience": ["a backend engineer", "a new Python user"],
    }
).with_columns(
    pl.format("Write for {}.", "audience").alias("instructions")
)

result = requests.with_columns(
    pl.col("prompt")
    .llm.openai(
        model="gpt-4o-mini",
        system=pl.col("instructions"),
    )
    .alias("explanation")
)
```

## Structured extraction

Pass a Pydantic model class as `schema=` when downstream code needs typed fields instead of free-form text:

```python
from typing import Literal

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    category: Literal["returns", "billing", "shipping", "other"]
    urgency: int = Field(ge=1, le=5)
    summary: str


messages = pl.DataFrame(
    {
        "message": [
            "The replacement still has not arrived and I need it tomorrow.",
            "Please send a VAT invoice for order 1234.",
        ]
    }
)

result = (
    messages.with_columns(
        pl.col("message")
        .llm.openai(
            model="gpt-4o-mini",
            schema=Ticket,
            system="Extract support-ticket fields.",
        )
        .alias("ticket")
    )
    .unnest("ticket")
)
```

`result` has `category`, `urgency`, and `summary` columns with Polars dtypes derived from the structured response.

## Concurrent calls

Async-prefixed methods process rows concurrently. Use `max_concurrency` to limit simultaneous provider requests:

```python
result = messages.with_columns(
    pl.col("message")
    .llm.aopenai(
        model="gpt-4o-mini",
        max_concurrency=10,
    )
    .alias("answer")
)
```

Despite the method name, the expression is used like any other Polars expression; you do not need to write `await`. The library manages the async calls while Polars evaluates the batch.

!!! tip

    Start with a conservative concurrency limit and raise it only when the provider's rate limits and your account capacity allow it.

## Retries and error metadata

For batch jobs, metadata mode keeps successful rows and records per-row failures:

```python
result = messages.with_columns(
    pl.col("message")
    .llm.aopenai(
        model="gpt-4o-mini",
        max_concurrency=10,
        retries=3,
        backoff=0.5,
        cache=True,
        with_metadata=True,
    )
    .alias("llm_result")
).unnest("llm_result")

failed = result.filter(pl.col("error").is_not_null())
```

The chat metadata struct contains `content`, `elapsed_ms`, and `error`. `cache=True` deduplicates identical inputs within that evaluated batch. It is not a persistent cache between separate DataFrame evaluations.

If a failure should stop the entire operation, use `on_error="raise"` instead. Without metadata, the default `on_error="null"` returns null for failed rows and emits a warning.

## Embeddings and cosine similarity

Create embeddings in batches with `chunk_size`. Supplying `dim` produces a fixed-size Polars `Array` and catches unexpected dimension changes:

```python
documents = pl.DataFrame(
    {
        "text": [
            "Polars is a DataFrame library written in Rust.",
            "Embeddings map text to numeric vectors.",
            "A sourdough starter contains wild yeast.",
        ]
    }
).with_columns(
    pl.col("text")
    .llm.openai_embed(
        model="text-embedding-3-small",
        chunk_size=64,
        dim=1536,
    )
    .alias("vector")
)
```

Compare every vector with a literal vector or another vector column. This calculation is native Polars and does not make a provider call:

```python
reference_vector = documents.get_column("vector")[0].to_list()

scored = documents.with_columns(
    pl.col("vector").llm.cosine(reference_vector).alias("similarity")
).sort("similarity", descending=True)
```

A literal vector must have the same dimension as the vectors in the column.

## Embeddings and nearest-neighbor search

For top-K matching, embed the query and corpus with the same model, then join them through `.ann.knn`:

```python
queries = pl.DataFrame({"query": ["fast tabular data processing"]}).with_columns(
    pl.col("query")
    .llm.openai_embed(model="text-embedding-3-small", dim=1536)
    .alias("vector")
)

matches = queries.ann.knn(
    documents,
    on="vector",
    k=2,
    metric="cosine",
    backend="brute",
)

print(matches.select("query", "text", "rank", "score"))
```

Lower `score` values are closer matches. The brute-force backend requires the `[ann]` extra and is a good default for smaller corpora. `backend="auto"` can switch to usearch for a corpus of roughly 50,000 rows or more when usearch is installed.

Use `flat=False` to retain one row per query with a `neighbors: List[Struct]` column:

```python
nested_matches = queries.ann.knn(
    documents,
    on="vector",
    k=2,
    flat=False,
)
```

## TypeSafe decisions

Evaluate several decision primitives in one request. The source expression becomes the TypeSafe state:

```python
from typesafe_sdk import Choice, Noul, Score


questions = {
    "department": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "returns": "Exchanges, wrong items, or damaged items",
            "shipping": "Delivery status, delays, or lost packages",
            "billing": "Charges, invoices, or payment problems",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this request?",
        criteria=["Can wait", "Needs attention today", "Customer is blocked"],
    ),
    "needs_reply": Noul(instructions="Does the customer need a reply?"),
}

decisions = messages.with_columns(
    pl.col("message").llm.typesafe(questions=questions).alias("decision")
).unnest("decision")
```

Use `atypesafe(..., max_concurrency=10)` for bounded concurrent evaluation.

## Bring your own LangChain client

Pass a configured client to reuse provider settings or target a compatible endpoint. When `client=` is supplied, `model=` is optional:

```python
from langchain_openai import ChatOpenAI


client = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,
    max_retries=2,
)

result = messages.with_columns(
    pl.col("message").llm.openai(client=client).alias("answer")
)
```
