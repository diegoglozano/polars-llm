# TODO — Feature Ideas

A running list of features to consider for `polars-llm`. Ordered roughly by value vs. effort.

## ✅ Done

- `.llm` expression namespace registered on Polars expressions.
- Sync + async chat verbs for OpenAI, Anthropic, and Gemini (`openai`/`aopenai`, `anthropic`/`aanthropic`, `gemini`/`agemini`).
- Sync + async embedding verbs for OpenAI and Gemini (`openai_embed`/`aopenai_embed`, `gemini_embed`/`agemini_embed`).
- Per-row prompts and per-row system messages (literal string or `pl.Expr`).
- Structured outputs via `schema=MyPydanticModel`, backed by LangChain's `with_structured_output`.
- Retries with exponential backoff (`retries=`, `backoff=`).
- In-batch caching: `cache=True` memoizes identical `(prompt, system)` (chat) or `text` (embed) keys within a batch.
- Async concurrency cap via `asyncio.Semaphore` (`max_concurrency=`).
- Per-row metadata struct (`with_metadata=True` → `{content, elapsed_ms, error}` for chat, `{vector, dim, elapsed_ms, error}` for embed).
- Configurable error handling: `on_error="null" | "raise"` with a `UserWarning` when failures are silently nulled.
- Optional provider extras: `polars-llm[openai]`, `[anthropic]`, `[gemini]`, `[all]` so the base install stays light.
- Bring-your-own client: pass any preconfigured LangChain chat / embeddings instance via `client=` to skip the in-tree constructor.
- `**model_kwargs` pass-through (temperature, max_tokens, timeout, base_url, …).

## Remaining

### More providers

- **Ollama (local)** — `ChatOllama` and `OllamaEmbeddings` from `langchain-ollama`. Big win for offline / dev workflows. Tiny add: register `ollama`/`aollama`/`ollama_embed`/… in `_make_chat`/`_make_embed` and a new `[ollama]` extra.
- **AWS Bedrock** — `ChatBedrockConverse` and `BedrockEmbeddings` from `langchain-aws`. Same pattern, plus a note about IAM auth.
- **Azure OpenAI** — already reachable via `client=AzureChatOpenAI(...)`, but a first-class `azure_openai` verb that forwards `azure_endpoint`/`api_version`/`deployment_name` would be friendlier.
- **Mistral** (`langchain-mistralai`), **Cohere** (`langchain-cohere`), **HuggingFace** (`langchain-huggingface`).
- **Voyage AI embeddings** — closes the Anthropic-side gap (Anthropic recommends Voyage for embeddings) via `langchain-voyageai`.

### Multimodal inputs

- Accept image columns (URL, bytes, file path, or `Struct{type, data}`) on chat verbs that support vision (`gpt-4o`, Claude Sonnet 4.6, Gemini 2.5). Roughly: `pl.col("image_url").llm.openai(model="gpt-4o", prompt=pl.col("question"))` — promote prompt to a kwarg when the column is non-text. Requires a modest reshape of `_input_struct` to encode multipart content blocks.

### Tool / function calling

- `tools=[...]` plumbed through `model.bind_tools(...)`. Return either the raw tool-call struct or a Utf8 column of executed-tool results when `tool_executor=` is provided. Useful for agent-style enrichment loops on a DataFrame.

### Generic LangChain runnable

- `pl.col("prompt").llm.invoke(runnable=my_lcel_chain)` — accept any LangChain `Runnable` (chain, agent, retriever, …). Lets users compose `with_structured_output` + `RunnableWithFallbacks` + retrieval pipelines and apply them per row without leaving Polars.

### Persistent caching

- File-backed cache (sqlite or `diskcache`) keyed on the same tuple as the in-batch cache, so repeated runs of the same DataFrame don't pay the model bill twice. Surface as `cache="path/to/cache.sqlite"` in addition to the current `cache=True`.

### Adaptive rate limiting

- Token-bucket / leaky-bucket limiter that respects per-provider RPM/TPM tiers, honours `Retry-After` from 429s, and adapts on the fly. Extends `max_concurrency` from a hard cap to a smarter governor.

### Token / cost accounting

- Add `tokens: Struct{input: Int64, output: Int64}` and `cost_usd: Float64` to the metadata struct (read from LangChain's `usage_metadata`). Pair with a `df["meta"].struct.field("cost_usd").sum()` example in docs so users can budget at a glance.

### Lifecycle / observability hooks

- `on_request` / `on_response` callbacks like `polars-api` has, plus a `LANGSMITH_TRACING=1`-aware passthrough (the LangChain runtime already supports this; just need to make sure our `map_batches` invocation doesn't break the trace context).

### Prompt-template sugar

- `pl.col("text").llm.openai(model=..., template="Translate to {lang}: {text}", lang=pl.col("lang"))` — sugar on top of `pl.format`, but with named placeholders. Reduces boilerplate when prompts are templated.
- Few-shot helper: `examples=[("input", "output"), ...]` prepends a few-shot block automatically.

### Native batching

- Today every row is a separate `invoke`/`ainvoke` call so we can capture per-row errors. Add an opt-in `batch_size=N` that uses `model.batch()` / `model.abatch()` in chunks for providers that bill or rate-limit on requests rather than tokens. On batch failure, fall back to per-row to preserve error isolation.

### Embedding ergonomics

- **Chunked embeddings** — `embed_documents` accepts a list, so a `chunk_size=` knob would issue one API call per N rows for ~10× cheaper embedding generation.
- **Similarity helpers** — `pl.col("vector_a").llm.cosine(pl.col("vector_b"))` sugar that lowers to a Polars expression (no provider call).
- **ANN join** — `df.llm.knn(other, on="vector", k=5)` to do a top-K join against another DataFrame of embeddings, backed by either brute-force or `hnswlib`.

### Default-model configuration

- `POLARS_LLM_DEFAULT_OPENAI_MODEL` etc. so `pl.col("p").llm.openai()` works without `model=` once configured. Useful in notebooks.

### Tests / benchmarks

- A `benchmarks/` script (mirroring `polars-api/benchmarks/bench.py`) that spins up a fake LangChain server locally and times sync vs. async vs. native batched throughput, reporting rps / cost.

### Docs

- A "RAG over a DataFrame" notebook example: build embeddings on one column, KNN-join against a query DataFrame, then call a chat verb with the retrieved context.
- A "structured extraction" tutorial extracting fields from unstructured text into a typed DataFrame.

### Multipart / file uploads

- Mostly relevant for OpenAI's audio/transcription endpoints. Likely belongs in a sibling `polars-audio` namespace rather than here — flag and defer.
