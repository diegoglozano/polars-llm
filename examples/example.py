"""End-to-end usage of polars-llm.

Set OPENAI_API_KEY (or ANTHROPIC_API_KEY / GOOGLE_API_KEY) before running.
"""

import polars as pl
from pydantic import BaseModel

import polars_llm  # noqa: F401  — registers the `.llm` namespace


class Sentiment(BaseModel):
    label: str  # "positive" | "neutral" | "negative"
    confidence: float


df = pl.DataFrame({
    "user_prompt": [
        "Polars is amazing!",
        "I had a terrible day debugging.",
        "It's just another release.",
    ],
})

print(
    df.with_columns(
        # 1) plain chat completion
        pl.col("user_prompt").llm.openai(model="gpt-4o-mini").alias("answer"),
        # 2) structured output via Pydantic
        pl.col("user_prompt")
        .llm.openai(model="gpt-4o-mini", schema=Sentiment, system="Score sentiment for the text.")
        .alias("sentiment"),
        # 3) async chat with concurrency cap
        pl.col("user_prompt").llm.aanthropic(model="claude-sonnet-4-6", max_concurrency=4).alias("claude"),
        # 4) embeddings
        pl.col("user_prompt").llm.openai_embed(model="text-embedding-3-small").alias("vector"),
    ),
)
