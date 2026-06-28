"""Native token-counting accelerator for polars-llm (optional).

If this package is importable, `polars_llm` routes its token-counting verbs
through these in-engine Polars expressions (no Python UDF). Otherwise it falls
back to the pure-Python path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

_PLUGIN_PATH = Path(__file__).parent

__all__ = ["count_gemma", "count_openai"]


def count_openai(expr: pl.Expr, encoding: str) -> pl.Expr:
    """Count OpenAI tokens natively via tiktoken-rs. Returns an Int64 expression."""
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="count_openai",
        args=expr,
        kwargs={"encoding": encoding},
        is_elementwise=True,
    )


def count_gemma(expr: pl.Expr, tokenizer_path: str) -> pl.Expr:
    """Count Gemini/Gemma tokens natively via the Gemma tokenizer.json. Returns Int64."""
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="count_gemma",
        args=expr,
        kwargs={"path": tokenizer_path},
        is_elementwise=True,
    )
