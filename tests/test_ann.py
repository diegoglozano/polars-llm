"""Tests for the polars_llm `.ann` namespace (top-K nearest-neighbour join)."""

from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest

import polars_llm  # noqa: F401  registers the `.ann` namespace
from polars_llm import _ann

usearch = pytest.importorskip("usearch.index", reason="usearch not installed")


def _docs() -> pl.DataFrame:
    return pl.DataFrame({
        "doc_id": ["a", "b", "c", "d"],
        "vector": [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ],
    })


def _queries() -> pl.DataFrame:
    return pl.DataFrame({
        "q_id": ["q1", "q2"],
        "vector": [
            [0.9, 0.1],
            [0.0, 1.0],
        ],
    })


# --------------------------------------------------------------------------
# Brute-force backend
# --------------------------------------------------------------------------
class TestBrute:
    def test_flat_join_shape_and_top1(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=2, backend="brute")

        assert out.height == 2 * 2
        assert "rank" in out.columns
        assert "score" in out.columns
        assert "vector_right" in out.columns  # collision suffix

        first = out.filter(pl.col("q_id") == "q1").sort("rank")
        assert first.get_column("rank").to_list() == [0, 1]
        assert first.get_column("doc_id").to_list()[0] == "a"

        second = out.filter(pl.col("q_id") == "q2").sort("rank")
        assert second.get_column("doc_id").to_list()[0] == "b"
        assert math.isclose(second.get_column("score")[0], 0.0, abs_tol=1e-6)

    def test_left_right_on(self) -> None:
        left = _queries().rename({"vector": "qv"})
        right = _docs().rename({"vector": "dv"})
        out = left.ann.knn(right, left_on="qv", right_on="dv", k=1, backend="brute")
        assert out.height == 2
        assert out.get_column("doc_id").to_list() == ["a", "b"]

    def test_k_clamped_to_corpus(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=99, backend="brute")
        assert out.height == 2 * 4

    def test_metric_l2(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=1, metric="l2", backend="brute")
        # q2 = (0, 1) → exact match for doc "b" → squared distance 0
        q2 = out.filter(pl.col("q_id") == "q2")
        assert q2.get_column("doc_id").to_list() == ["b"]
        assert math.isclose(q2.get_column("score")[0], 0.0, abs_tol=1e-6)

    def test_metric_ip(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=1, metric="ip", backend="brute")
        # ip distance = 1 - dot. q1=(0.9,0.1): doc "c"=(1,1) → dot=1.0 (largest), so closest under ip.
        q1 = out.filter(pl.col("q_id") == "q1")
        assert q1.get_column("doc_id").to_list() == ["c"]

    def test_array_dtype(self) -> None:
        docs = _docs().with_columns(pl.col("vector").cast(pl.Array(pl.Float32, 2)))
        queries = _queries().with_columns(pl.col("vector").cast(pl.Array(pl.Float32, 2)))
        out = queries.ann.knn(docs, on="vector", k=1, backend="brute")
        assert out.get_column("doc_id").to_list() == ["a", "b"]

    def test_nested_output(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=2, backend="brute", flat=False)
        assert out.height == 2
        assert "neighbors" in out.columns
        first = out.filter(pl.col("q_id") == "q1").get_column("neighbors").to_list()[0]
        assert len(first) == 2
        assert first[0]["rank"] == 0
        assert first[0]["doc_id"] == "a"

    def test_dim_mismatch_raises(self) -> None:
        bad = pl.DataFrame({"doc_id": ["x"], "vector": [[1.0, 2.0, 3.0]]})
        with pytest.raises(ValueError, match="dim mismatch"):
            _queries().ann.knn(bad, on="vector", k=1, backend="brute")

    def test_empty_other_raises(self) -> None:
        empty = pl.DataFrame(
            {"doc_id": [], "vector": []},
            schema={"doc_id": pl.Utf8, "vector": pl.List(pl.Float64)},
        )
        with pytest.raises(ValueError, match="empty"):
            _queries().ann.knn(empty, on="vector", k=1, backend="brute")

    def test_bad_dtype_raises(self) -> None:
        bad = pl.DataFrame({"doc_id": ["x"], "vector": ["not a vector"]})
        with pytest.raises(TypeError, match="List\\[Float"):
            _queries().ann.knn(bad, on="vector", k=1, backend="brute")

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _queries().ann.knn(_docs(), on="missing", k=1, backend="brute")

    def test_on_and_left_on_conflict_raises(self) -> None:
        with pytest.raises(ValueError, match="either `on`"):
            _queries().ann.knn(_docs(), on="vector", left_on="vector", k=1, backend="brute")

    def test_k_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="`k`"):
            _queries().ann.knn(_docs(), on="vector", k=0, backend="brute")

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            _queries().ann.knn(_docs(), on="vector", k=1, metric=cast("str", "manhattan"), backend="brute")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            _queries().ann.knn(_docs(), on="vector", k=1, backend=cast("str", "faiss"))


# --------------------------------------------------------------------------
# usearch backend (skipped automatically if not installed via importorskip above)
# --------------------------------------------------------------------------
class TestUsearch:
    def test_returns_same_top1_as_brute(self) -> None:
        kw = {"on": "vector", "k": 1}
        brute = _queries().ann.knn(_docs(), backend="brute", **kw)
        ann = _queries().ann.knn(_docs(), backend="usearch", **kw)
        assert brute.get_column("doc_id").to_list() == ann.get_column("doc_id").to_list()

    def test_l2_metric(self) -> None:
        out = _queries().ann.knn(_docs(), on="vector", k=1, metric="l2", backend="usearch")
        assert out.get_column("doc_id").to_list()[1] == "b"

    def test_auto_picks_brute_for_small(self) -> None:
        assert _ann._pick_backend("auto", n_right=10) == "brute"

    def test_auto_picks_usearch_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_ann, "_AUTO_USEARCH_THRESHOLD", 4)
        assert _ann._pick_backend("auto", n_right=4) == "usearch"


def test_auto_falls_back_when_usearch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ann, "_USEARCH", None)
    monkeypatch.setattr(_ann, "_AUTO_USEARCH_THRESHOLD", 1)
    assert _ann._pick_backend("auto", n_right=1_000_000) == "brute"


def test_explicit_usearch_without_extra_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ann, "_USEARCH", None)
    with pytest.raises(ImportError, match="ann"):
        _queries().ann.knn(_docs(), on="vector", k=1, backend="usearch")
