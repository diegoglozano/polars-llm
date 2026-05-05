"""DataFrame-level ``.ann`` namespace for top-K nearest-neighbour joins.

Backed by either pure-NumPy brute force (default) or `usearch` (HNSW with
optional scalar quantization). `usearch` is an optional extra:

    pip install polars-llm[ann]

Distance convention: **lower score = closer match**, mirroring `usearch`.

* ``cosine`` — ``1 - cos(a, b)`` ∈ [0, 2]
* ``ip``     — ``1 - dot(a, b)``  (true inner-product distance for normalised
  embeddings; for un-normalised vectors it is still monotone in similarity)
* ``l2``     — squared L2 distance ``||a - b||²``
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

Metric = Literal["cosine", "ip", "l2"]
Backend = Literal["auto", "brute", "usearch"]

# Above this many `other` rows, ``backend="auto"`` switches from brute-force
# NumPy to usearch (when usearch is installed).
_AUTO_USEARCH_THRESHOLD = 50_000

_USEARCH: Any = None
with contextlib.suppress(ImportError):  # pragma: no cover - import guard
    from usearch.index import Index as _USEARCH


def _require_usearch() -> Any:
    if _USEARCH is None:
        raise ImportError(
            "polars-llm: `backend='usearch'` requires the optional `ann` extra. "
            "Install it with `pip install polars-llm[ann]`.",
        )
    return _USEARCH


def _to_matrix(df: pl.DataFrame, col: str) -> np.ndarray:
    import numpy as np

    if col not in df.columns:
        raise ValueError(f"polars-llm: column {col!r} not found in DataFrame.")
    series = df.get_column(col)
    dtype = series.dtype
    if not (isinstance(dtype, (pl.List, pl.Array)) and dtype.inner in (pl.Float32, pl.Float64)):
        raise TypeError(
            f"polars-llm: column {col!r} must be List[Float32/64] or Array[Float32/64, dim]; got {dtype!r}.",
        )
    arr = np.asarray(series.to_list(), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"polars-llm: column {col!r} has variable-length or empty vectors; all rows must share one dimension.",
        )
    return np.ascontiguousarray(arr)


def _brute_search(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    metric: Metric,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    if metric == "cosine":
        q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - q @ c.T
    elif metric == "ip":
        dist = 1.0 - queries @ corpus.T
    elif metric == "l2":
        q_sq = (queries * queries).sum(axis=1, keepdims=True)
        c_sq = (corpus * corpus).sum(axis=1)
        dist = q_sq + c_sq - 2.0 * (queries @ corpus.T)
        np.maximum(dist, 0.0, out=dist)
    else:
        raise ValueError(f"polars-llm: unknown metric {metric!r}; expected one of 'cosine', 'ip', 'l2'.")

    n_corpus = corpus.shape[0]
    k_eff = min(k, n_corpus)
    if k_eff < n_corpus:
        part = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
    else:
        idx = np.argsort(dist, axis=1)
    scores = np.take_along_axis(dist, idx, axis=1)
    return idx.astype(np.uint32), scores.astype(np.float32)


_USEARCH_METRIC = {"cosine": "cos", "ip": "ip", "l2": "l2sq"}


def _usearch_search(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    metric: Metric,
    **index_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    Index = _require_usearch()
    dim = corpus.shape[1]
    index = Index(ndim=dim, metric=_USEARCH_METRIC[metric], **index_kwargs)
    index.add(np.arange(corpus.shape[0], dtype=np.uint64), corpus)
    matches = index.search(queries, count=min(k, corpus.shape[0]))
    keys = np.asarray(matches.keys, dtype=np.uint32)
    distances = np.asarray(matches.distances, dtype=np.float32)
    if keys.ndim == 1:
        keys = keys[None, :]
        distances = distances[None, :]
    return keys, distances


@pl.api.register_dataframe_namespace("ann")
class Ann:
    """DataFrame namespace for approximate / exact nearest-neighbour joins."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def knn(
        self,
        other: pl.DataFrame,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        k: int = 5,
        metric: Metric = "cosine",
        backend: Backend = "auto",
        flat: bool = True,
        suffix: str = "_right",
        rank_name: str = "rank",
        score_name: str = "score",
        **backend_kwargs: Any,
    ) -> pl.DataFrame:
        """Top-K nearest-neighbour join against ``other``.

        Each row in ``self`` is matched against ``other`` using the chosen
        ``metric``; the result holds the ``k`` closest neighbours per row.

        Parameters
        ----------
        other:
            The right-hand DataFrame to search.
        on, left_on, right_on:
            Vector column name(s). Use ``on="vector"`` when both DataFrames
            share the column name, otherwise pass ``left_on`` and ``right_on``.
        k:
            Number of neighbours to return. Clamped to ``len(other)``.
        metric:
            ``"cosine"`` (default), ``"ip"``, or ``"l2"``. Lower score = closer.
        backend:
            ``"auto"`` (brute force up to a few x 10^4 right rows, otherwise
            usearch when installed), ``"brute"``, or ``"usearch"``.
        flat:
            ``True`` (default) returns a flat join of ``len(self) * k`` rows
            with all columns from both sides plus ``rank`` / ``score``.
            ``False`` returns ``len(self)`` rows with a ``neighbors``
            ``List[Struct]`` column carrying the right-side rows.
        suffix:
            Collision suffix for right-side columns (only used when ``flat``).
        rank_name, score_name:
            Names of the rank/score columns added to the output.
        **backend_kwargs:
            Forwarded to ``usearch.index.Index`` (e.g. ``connectivity``,
            ``expansion_add``, ``expansion_search``, ``dtype``).
        """
        import numpy as np  # noqa: F401  -- ensures numpy is installed

        left_col, right_col = _resolve_keys(on, left_on, right_on)
        left_vec = _to_matrix(self._df, left_col)
        right_vec = _to_matrix(other, right_col)
        if left_vec.shape[1] != right_vec.shape[1]:
            raise ValueError(
                f"polars-llm: vector dim mismatch: left {left_vec.shape[1]} != right {right_vec.shape[1]}.",
            )
        if other.height == 0:
            raise ValueError("polars-llm: `other` is empty; cannot run knn.")
        if k < 1:
            raise ValueError(f"polars-llm: `k` must be >= 1, got {k}.")

        chosen = _pick_backend(backend, other.height)
        if chosen == "brute":
            indices, scores = _brute_search(left_vec, right_vec, k, metric)
        else:
            indices, scores = _usearch_search(left_vec, right_vec, k, metric, **backend_kwargs)

        return _assemble(
            self._df,
            other,
            indices,
            scores,
            flat=flat,
            suffix=suffix,
            rank_name=rank_name,
            score_name=score_name,
        )


def _resolve_keys(
    on: str | None,
    left_on: str | None,
    right_on: str | None,
) -> tuple[str, str]:
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("polars-llm: pass either `on` or (`left_on`, `right_on`), not both.")
        return on, on
    if left_on is None or right_on is None:
        raise ValueError("polars-llm: provide `on=` or both `left_on=` and `right_on=`.")
    return left_on, right_on


def _pick_backend(backend: Backend, n_right: int) -> Literal["brute", "usearch"]:
    if backend == "brute":
        return "brute"
    if backend == "usearch":
        _require_usearch()
        return "usearch"
    if backend == "auto":
        if n_right >= _AUTO_USEARCH_THRESHOLD and _USEARCH is not None:
            return "usearch"
        return "brute"
    raise ValueError(f"polars-llm: unknown backend {backend!r}; expected 'auto', 'brute', or 'usearch'.")


def _assemble(
    left: pl.DataFrame,
    right: pl.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    flat: bool,
    suffix: str,
    rank_name: str,
    score_name: str,
) -> pl.DataFrame:
    import numpy as np

    n_left, k_eff = indices.shape
    left_idx_col = "__polars_llm_left_idx"
    right_idx_col = "__polars_llm_right_idx"

    pairs = pl.DataFrame({
        left_idx_col: np.repeat(np.arange(n_left, dtype=np.uint32), k_eff),
        right_idx_col: indices.reshape(-1).astype(np.uint32),
        rank_name: np.tile(np.arange(k_eff, dtype=np.uint32), n_left),
        score_name: scores.reshape(-1).astype(np.float32),
    })

    right_indexed = right.with_row_index(right_idx_col)
    pairs_with_right = pairs.join(right_indexed, on=right_idx_col).drop(right_idx_col)

    if flat:
        left_indexed = left.with_row_index(left_idx_col)
        return left_indexed.join(pairs_with_right, on=left_idx_col, suffix=suffix).drop(left_idx_col)

    neighbor_cols = [c for c in pairs_with_right.columns if c != left_idx_col]
    nested = pairs_with_right.group_by(left_idx_col, maintain_order=True).agg(
        pl.struct(neighbor_cols).alias("neighbors"),
    )
    return left.with_row_index(left_idx_col).join(nested, on=left_idx_col, how="left").drop(left_idx_col)
