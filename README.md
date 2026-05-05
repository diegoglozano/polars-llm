# polars-llm

[![PyPI version](https://img.shields.io/pypi/v/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Python versions](https://img.shields.io/pypi/pyversions/polars-llm.svg)](https://pypi.org/project/polars-llm/)
[![Build status](https://img.shields.io/github/actions/workflow/status/diegoglozano/polars-llm/main.yml?branch=main)](https://github.com/diegoglozano/polars-llm/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/diegoglozano/polars-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/diegoglozano/polars-llm)
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

- **Repository**: <https://github.com/diegoglozano/polars-llm>
- **Documentation**: <https://diegoglozano.github.io/polars-llm/>
- **PyPI**: <https://pypi.org/project/polars-llm/>

---

## Why polars-llm?

- **Expression-native** — works inside `with_columns`, `select`, and any other Polars expression context. No Python `for` loops over rows, no notebook glue.
- **Sync and async** — every provider verb has an `a`-prefixed async sibling that fans out concurrently with `asyncio.gather` and an optional `max_concurrency` cap.
- **Per-row prompts and system messages** — both the prompt and the system message can be Polars expressions, so you can build them from other columns.
- **Structured outputs** — pass a Pydantic model as `schema=` to get a struct column back, parsed via LangChain's `with_structured_output`.
- **Embeddings, too** — `openai_embed` and `gemini_embed` return `List[Float64]` columns ready for vector search.
- **Top-K nearest-neighbour join** — `df.ann.knn(other, on="vector", k=5)` joins one DataFrame of embeddings against another, with a brute-force NumPy default and an optional [`usearch`](https://github.com/unum-cloud/usearch) HNSW backend for larger corpora.
- **Powered by [LangChain](https://python.langchain.com/)** — you get the same retries, batching, and observability primitives the rest of the LangChain ecosystem uses, plumbed straight into a DataFrame.

Common use cases:

- Summarise, classify, translate, or extract structured fields from a column of text.
- Score rows against a custom rubric using an LLM-as-judge.
- Build embeddings for a corpus directly from a DataFrame, ready to write to a vector database.
- Mix LLM calls with the rest of your pipeline (joins, filters, group-bys) without leaving Polars.

## Installation

`polars-llm` keeps its base install light. Pick the providers you need as extras:

```sh
# Just one provider
pip install "polars-llm[openai]"
pip install "polars-llm[anthropic]"
pip install "polars-llm[gemini]"

# Top-K nearest-neighbour joins (adds usearch + numpy)
pip install "polars-llm[ann]"

# Or all of them
pip install "polars-llm[all]"

# uv
uv add "polars-llm[all]"
```

Requires Python 3.9+ and Polars 1.0+.

Authentication follows LangChain conventions — set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` in your environment before importing.

## Quickstart

### 1. Chat completion per row

```python
import polars as pl
import polars_llm  # noqa: F401

df = (
    pl.DataFrame({"user_prompt": [
        "What is the capital of Spain?",
        "What is the capital of France?",
    ]})
    .with_columns(
        pl.col("user_prompt").llm.openai(model="gpt-4o-mini").alias("answer")
    )
)
```

### 2. System prompt — literal or per-row

```python
# Same system prompt for every row
pl.col("user_prompt").llm.anthropic(
    model="claude-sonnet-4-6",
    system="Answer in fewer than 10 words.",
)

# Per-row system prompt from another column
pl.col("user_prompt").llm.gemini(
    model="gemini-2.5-pro",
    system=pl.col("system_prompt"),
)
```

### 3. Async for throughput

The `a`-prefixed verbs run concurrently across the batch, capped at `max_concurrency`:

```python
df.with_columns(
    pl.col("user_prompt").llm.aopenai(
        model="gpt-4o-mini",
        max_concurrency=20,
    ).alias("answer")
)
```

### 4. Structured output with Pydantic

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str  # "positive" | "neutral" | "negative"
    confidence: float

df.with_columns(
    pl.col("review").llm.openai(
        model="gpt-4o-mini",
        schema=Sentiment,
    ).alias("sentiment")
).unnest("sentiment")
```

### 5. Embeddings

```python
df.with_columns(
    pl.col("text").llm.openai_embed(
        model="text-embedding-3-small",
    ).alias("vector")
)
```

### 6. Top-K nearest-neighbour join

Once you have an embedding column on each side, `df.ann.knn` returns the `k` closest rows from `other` for every row of `df`:

```python
import polars as pl
import polars_llm  # noqa: F401  — registers the `.ann` namespace

queries = pl.DataFrame({
    "q_id": ["q1", "q2"],
    "vector": [[0.9, 0.1], [0.0, 1.0]],
})
docs = pl.DataFrame({
    "doc_id": ["a", "b", "c"],
    "vector": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
})

queries.ann.knn(docs, on="vector", k=2)
# shape: (4, 5)
# ┌──────┬───────────┬────────┬──────┬──────────┐
# │ q_id ┆ vector    ┆ doc_id ┆ rank ┆ score    │
# ╞══════╪═══════════╪════════╪══════╪══════════╡
# │ q1   ┆ [0.9,0.1] ┆ a      ┆ 0    ┆ 0.005…   │
# │ q1   ┆ [0.9,0.1] ┆ c      ┆ 1    ┆ 0.071…   │
# │ q2   ┆ [0.0,1.0] ┆ b      ┆ 0    ┆ 0.0      │
# │ q2   ┆ [0.0,1.0] ┆ c      ┆ 1    ┆ 0.293…   │
# └──────┴───────────┴────────┴──────┴──────────┘
```

`backend="auto"` (default) uses brute-force NumPy under ~50k rows and switches to `usearch` HNSW for larger corpora when the `[ann]` extra is installed. Force one with `backend="brute"` or `backend="usearch"`. Pass `flat=False` to get a `neighbors: List[Struct]` column instead of a flat join. Lower `score` = closer match.

### 7. Retries, caching, metadata

```python
pl.col("user_prompt").llm.aanthropic(
    model="claude-sonnet-4-6",
    retries=3,
    backoff=0.5,
    max_concurrency=10,
    cache=True,            # dedupe identical prompts within a batch
    with_metadata=True,    # struct {content, elapsed_ms, error}
)
```

## API reference

All methods live under the `.llm` namespace on any Polars expression that resolves to a string column.

### Chat verbs

| Method                     | Provider      | Mode         |
| -------------------------- | ------------- | ------------ |
| `openai` / `aopenai`       | OpenAI        | sync / async |
| `anthropic` / `aanthropic` | Anthropic     | sync / async |
| `gemini` / `agemini`       | Google Gemini | sync / async |

### Embedding verbs

| Method                           | Provider          | Mode         |
| -------------------------------- | ----------------- | ------------ |
| `openai_embed` / `aopenai_embed` | OpenAI Embeddings | sync / async |
| `gemini_embed` / `agemini_embed` | Google Gemini     | sync / async |

> Anthropic does not currently offer a first-party embeddings API.

### DataFrame `.ann` namespace

`df.ann.knn(other, **kwargs)` — top-K nearest-neighbour join between two DataFrames of vectors.

| Argument                    | Default            | Notes                                                                                                 |
| --------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `on` / `left_on`/`right_on` | —                  | Vector column name(s). Use `on=` when both sides share a name, otherwise both `*_on`.                 |
| `k`                         | `5`                | Number of neighbours per row. Clamped to `len(other)`.                                                |
| `metric`                    | `"cosine"`         | One of `"cosine"`, `"ip"`, `"l2"` (squared L2). Lower score = closer match.                           |
| `backend`                   | `"auto"`           | `"auto"` switches to `usearch` above ~50k right rows when installed; otherwise `"brute"`.             |
| `flat`                      | `True`             | `True` → `len(df) * k` rows. `False` → one row per query with a `List[Struct]` `neighbors` col.       |
| `suffix`                    | `"_right"`         | Right-side column collision suffix (flat output only).                                                |
| `rank_name`/`score_name`    | `"rank"`/`"score"` | Names of the added rank and distance columns.                                                         |
| `**backend_kwargs`          | —                  | Forwarded to `usearch.index.Index` (`connectivity`, `expansion_add`, `expansion_search`, `dtype`, …). |

The vector columns must be `List[Float32/64]` or `Array[Float32/64, dim]`, and dimensions must match between the two DataFrames.

### Common arguments

All verbs are keyword-only and accept:

- **`model`** _(str)_ — model name forwarded to LangChain (e.g. `"gpt-4o-mini"`, `"claude-sonnet-4-6"`, `"gemini-2.5-pro"`).
- **`system`** _(chat only)_ — literal string or `pl.Expr` for a per-row system prompt.
- **`schema`** _(chat only)_ — a Pydantic model class. Returns a struct column with the schema fields, via `with_structured_output`.
- **`client`** — a pre-configured LangChain chat or embeddings instance (skips the in-tree constructor and is handy for advanced configuration like custom base URLs).
- **`retries`** _(int, default 0)_ — retry on any exception raised by the provider call.
- **`backoff`** _(float, default 0.0)_ — exponential backoff base (seconds).
- **`max_concurrency`** _(async only, int)_ — cap on in-flight requests via `asyncio.Semaphore`.
- **`cache`** _(bool, default False)_ — memoise identical inputs within a batch.
- **`with_metadata`** _(bool, default False)_ — return a struct column with timing and error metadata instead of just the content / vector.
- **`on_error`** _("null" | "raise", default "null")_ — when `with_metadata=False`, what to do on errors. `"null"` replaces failures with `None` and emits a warning; `"raise"` re-raises immediately.
- **`**model_kwargs`** — any additional keyword arguments forwarded to the underlying LangChain class (e.g. `temperature=`, `max_tokens=`, `timeout=`).

### Return types

| Mode                 | Default dtype                             | With `with_metadata=True`                                                     |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------------------------- |
| Chat (no `schema`)   | `Utf8`                                    | `Struct{content: Utf8, elapsed_ms: Float64, error: Utf8}`                     |
| Chat (with `schema`) | `Struct{...}` matching the Pydantic model | Same struct; content JSON-serialised under `content`                          |
| Embeddings           | `List[Float64]`                           | `Struct{vector: List[Float64], dim: Int64, elapsed_ms: Float64, error: Utf8}` |

## Tips and patterns

- **Build prompts from columns** with `pl.format("Translate to {}: {}", pl.col("language"), pl.col("text"))`.
- **Bring your own client** to share a single `ChatOpenAI` (with custom `base_url`, `organization`, etc.) across many calls — pass it as `client=`.
- **Watch the warning** — when a request fails and is silently nulled, polars-llm emits a `UserWarning` so you don't ship a column of nulls by accident. Pass `with_metadata=True` to inspect per-row errors instead.
- **Combine with lazy frames** — every verb is an expression, so it composes inside `LazyFrame.with_columns(...)`.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md). Please open an issue before starting on larger changes.

## License

[MIT](./LICENSE) © Diego Garcia Lozano

---

Inspired by and patterned after [polars-api](https://github.com/diegoglozano/polars-api).
