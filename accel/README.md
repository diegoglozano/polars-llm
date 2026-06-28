# polars-llm-accel

Optional native (Rust) accelerator for [`polars-llm`](../) token counting. It is
a [`pyo3-polars`](https://github.com/pola-rs/pyo3-polars) **expression plugin**:
the OpenAI and Gemini offline token counters become in-engine Polars expressions
(no Python UDF), powered by [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs)
and the Hugging Face [`tokenizers`](https://crates.io/crates/tokenizers) crate.

When this package is importable, `polars_llm`'s `openai_tokens` / `gemini_tokens`
verbs use it automatically; otherwise they fall back to the pure-Python paths.
It is intentionally a **separate distribution** so the base `polars-llm` install
stays pure-Python.

## ABI compatibility

A `pyo3-polars` plugin is compiled against a specific Polars version and must
match the installed `polars` wheel's ABI. This crate pins `polars = 0.53.0`,
which matches **polars-python 1.40.x**. If you upgrade Polars, bump the `polars`
/ `pyo3-polars` versions in `Cargo.toml` to the pair matching your Polars
release (see the `pyo3-polars` README / the `pyo3-polars/Cargo.toml` at the
relevant `py-<version>` tag of the polars repo) and rebuild.

## Build & install

Requires a Rust toolchain and [`maturin`](https://www.maturin.rs/).

```sh
# from this directory, into the active environment
maturin develop --release

# or build a wheel and install it
maturin build --release
pip install target/wheels/polars_llm_accel-*.whl   # or dist/ depending on --out
```

The wheel is `abi3` (one wheel works across CPython ≥ 3.9). The OpenAI vocab is
bundled in `tiktoken-rs`, so the OpenAI path needs no download. The Gemini path
loads a Gemma `tokenizer.json` from the path you pass (`tokenizer_path=` or
`POLARS_LLM_GEMMA_TOKENIZER`).

## Verify

```python
import polars as pl
import polars_llm  # noqa: F401

# Uses the native plugin automatically when this package is installed:
pl.DataFrame({"t": ["hello world"]}).with_columns(
    pl.col("t").llm.openai_tokens(model="gpt-4o").alias("n")
)
```
