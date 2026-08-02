"""Evaluate cross-encoder reranking over installed AtlasRAG hybrid candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch

import atlasrag
from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    ExactDenseRetriever,
    ReciprocalRankFusionRetriever,
    RetrievalQuery,
)

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MODEL_LICENSE = "Apache-2.0"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_LICENSE = "Apache-2.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else 0.0,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                rows[str(row["query-id"])][str(row["corpus-id"])] = score
    return dict(rows)


@dataclass(frozen=True, slots=True)
class Dataset:
    corpus: tuple[dict[str, Any], ...]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    eval_ids: tuple[str, ...]


def load_scifact(data_dir: Path) -> Dataset:
    corpus = tuple(load_jsonl(data_dir / "corpus.jsonl"))
    queries = {
        str(row["_id"]): str(row.get("text", ""))
        for row in load_jsonl(data_dir / "queries.jsonl")
    }
    qrels = load_qrels(data_dir / "qrels" / "test.tsv")
    return Dataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        eval_ids=tuple(sorted(qrels)),
    )


def row_text(row: dict[str, Any]) -> str:
    return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()


def metrics(
    ranked: Sequence[str], relevant: dict[str, int], *, top_k: int
) -> dict[str, float]:
    top = ranked[:top_k]
    relevant_ids = set(relevant)
    hits = [1 if item_id in relevant_ids else 0 for item_id in top]
    recall = sum(hits) / max(1, len(relevant_ids))
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, 1) if hit),
        0.0,
    )
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_count = min(top_k, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        f"recall@{top_k}": recall,
        f"mrr@{top_k}": reciprocal_rank,
        f"ndcg@{top_k}": dcg / ideal_dcg if ideal_dcg else 0.0,
        f"success@{top_k}": float(any(hits)),
    }


def mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one metric row is required")
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def select_latency_ids(
    query_ids: Sequence[str], *, seed: str, count: int
) -> tuple[str, ...]:
    ordered = sorted(
        query_ids,
        key=lambda query_id: (
            hashlib.sha256((seed + query_id).encode()).hexdigest(),
            query_id,
        ),
    )
    return tuple(ordered[: min(count, len(ordered))])


class FrozenSentenceTransformerEmbedding(EmbeddingModel):
    """Pinned MiniLM adapter with a checked corpus embedding cache."""

    def __init__(
        self,
        *,
        snapshot: Path,
        revision: str,
        cache_path: Path,
        batch_size: int,
        device: str,
    ) -> None:
        self._snapshot = snapshot.resolve()
        self._revision = revision
        self._cache_path = cache_path.resolve()
        self._batch_size = batch_size
        self._device = device
        self._model: Any | None = None
        self._dimension: int | None = None
        self.model_load_seconds = 0.0
        self.cache_load_seconds = 0.0
        self.python_vector_conversion_seconds = 0.0

    @property
    def model_id(self) -> str:
        return f"{DENSE_MODEL_NAME}@{self._revision}"

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            self._model = SentenceTransformer(
                str(self._snapshot),
                device=self._device,
                local_files_only=True,
            )
            self.model_load_seconds = time.perf_counter() - started
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = int(self._model.get_embedding_dimension())
            else:
                self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if not texts:
            return ()
        started = time.perf_counter()
        matrix = np.load(self._cache_path)
        self.cache_load_seconds = time.perf_counter() - started
        if matrix.ndim != 2 or matrix.shape != (len(texts), self.dimension):
            raise ValueError("cached corpus embedding shape does not match corpus")
        started = time.perf_counter()
        vectors = tuple(tuple(float(value) for value in row) for row in matrix)
        self.python_vector_conversion_seconds = time.perf_counter() - started
        return vectors

    def embed_query(self, text: str) -> Vector:
        model = self._load_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return tuple(float(value) for value in vector)


def build_chunks(
    dataset: Dataset,
) -> tuple[tuple[Chunk, ...], dict[str, str]]:
    chunks: list[Chunk] = []
    chunk_to_external: dict[str, str] = {}
    for ordinal, row in enumerate(dataset.corpus):
        external_id = str(row["_id"])
        text = row_text(row)
        document = Document.from_text(
            source_uri="dataset://scifact/corpus/" + quote(external_id, safe=""),
            text=text,
            document_id=external_id,
            metadata={
                "dataset.name": "scifact",
                "dataset.external_id": external_id,
            },
        )
        chunk = Chunk.from_document_span(
            document=document,
            start_char=0,
            end_char=len(text),
            ordinal=ordinal,
            strategy_id="beir-whole-document-v1",
        )
        chunks.append(chunk)
        chunk_to_external[chunk.chunk_id] = external_id
    if len(chunk_to_external) != len(chunks):
        raise ValueError("dataset external IDs must be unique")
    return tuple(chunks), chunk_to_external


def assert_installed_package(
    *, expected_version: str, forbidden_source_root: Path
) -> dict[str, str]:
    package_file = Path(atlasrag.__file__).resolve()
    if package_file.is_relative_to(forbidden_source_root.resolve()):
        raise RuntimeError(f"AtlasRAG imported from source tree: {package_file}")
    if "site-packages" not in package_file.as_posix():
        raise RuntimeError(
            f"AtlasRAG did not import from site-packages: {package_file}"
        )
    installed_version = metadata.version("atlasrag")
    if installed_version != expected_version:
        raise RuntimeError(
            f"installed version {installed_version} != expected {expected_version}"
        )
    return {
        "version": installed_version,
        "import_origin": "installed-wheel-site-packages",
        "module_relative_path": "atlasrag/__init__.py",
    }


def environment() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    try:
        cpu = subprocess.check_output(
            [
                "bash",
                "-lc",
                "lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2);print $2;exit}'",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "dependencies": {
            "atlasrag": version("atlasrag"),
            "numpy": version("numpy"),
            "sentence-transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
        },
    }


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def rerank_prefix(
    candidates: Sequence[Any], scores: Sequence[float], *, depth: int, top_k: int
) -> tuple[Any, ...]:
    prefix = list(zip(candidates[:depth], scores[:depth], strict=True))
    prefix.sort(
        key=lambda item: (
            -item[1],
            item[0].rank,
            item[0].chunk.chunk_id,
        )
    )
    return tuple(prefix[:top_k])


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    depths = tuple(sorted(set(args.candidate_depths)))
    if not depths or depths[0] < args.top_k:
        raise ValueError("candidate depths must be unique and >= top_k")
    max_depth = max(depths)
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(args.dataset_dir)
    if not args.embedding_cache.is_file():
        raise FileNotFoundError(args.embedding_cache)
    if not args.dense_model_snapshot.is_dir():
        raise FileNotFoundError(args.dense_model_snapshot)
    if not args.reranker_model_snapshot.is_dir():
        raise FileNotFoundError(args.reranker_model_snapshot)

    package = assert_installed_package(
        expected_version=args.expected_package_version,
        forbidden_source_root=args.forbid_source_root,
    )
    dataset = load_scifact(args.dataset_dir)
    eval_ids = (
        dataset.eval_ids[: args.query_limit] if args.query_limit else dataset.eval_ids
    )

    started = time.perf_counter()
    chunks, chunk_to_external = build_chunks(dataset)
    chunk_build_seconds = time.perf_counter() - started

    embedder = FrozenSentenceTransformerEmbedding(
        snapshot=args.dense_model_snapshot,
        revision=args.dense_model_revision,
        cache_path=args.embedding_cache,
        batch_size=args.embedding_batch_size,
        device=args.device,
    )
    hybrid = ReciprocalRankFusionRetriever(
        BM25Retriever(k1=args.bm25_k1, b=args.bm25_b),
        ExactDenseRetriever(embedder),
        rrf_k=args.rrf_k,
        candidate_k=args.hybrid_component_k,
    )
    started = time.perf_counter()
    hybrid.index(chunks)
    index_seconds = time.perf_counter() - started

    reranker = CrossEncoderReranker(
        RERANKER_MODEL_NAME,
        revision=args.reranker_model_revision,
        batch_size=args.reranker_batch_size,
        device=args.device,
        local_files_only=True,
    )
    if args.reranker_model_snapshot.resolve().name != args.reranker_model_revision:
        raise ValueError("reranker snapshot path does not match pinned revision")

    baseline_rows: list[dict[str, float]] = []
    reranked_rows: dict[int, list[dict[str, float]]] = {depth: [] for depth in depths}
    candidate_recall_rows: dict[int, list[float]] = {depth: [] for depth in depths}
    raw_rows: list[dict[str, Any]] = []

    for query_id in eval_ids:
        query_text = dataset.queries[query_id]
        relevant = dataset.qrels[query_id]
        candidates = hybrid.search(RetrievalQuery(text=query_text, top_k=max_depth))
        candidate_ids = [chunk_to_external[item.chunk.chunk_id] for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(f"duplicate hybrid candidate for query {query_id}")
        scores = reranker.score(query_text, [item.chunk for item in candidates])
        baseline_rows.append(metrics(candidate_ids, relevant, top_k=args.top_k))

        reranked_by_depth: dict[str, Any] = {}
        for depth in depths:
            prefix_ids = set(candidate_ids[:depth])
            candidate_recall = len(prefix_ids & set(relevant)) / max(1, len(relevant))
            candidate_recall_rows[depth].append(candidate_recall)
            ordered = rerank_prefix(
                candidates,
                scores,
                depth=depth,
                top_k=args.top_k,
            )
            ranked_ids = [chunk_to_external[item.chunk.chunk_id] for item, _ in ordered]
            row_metrics = metrics(ranked_ids, relevant, top_k=args.top_k)
            reranked_rows[depth].append(row_metrics)
            reranked_by_depth[str(depth)] = {
                "candidate_recall": candidate_recall,
                "metrics": row_metrics,
                "top": [
                    {
                        "external_id": chunk_to_external[item.chunk.chunk_id],
                        "chunk_id": item.chunk.chunk_id,
                        "reranker_score": score,
                        "candidate_rank": item.rank,
                        "candidate_score": item.score,
                        "candidate_score_kind": item.score_kind.value,
                        "candidate_method": item.method.value,
                        "source_uri": item.citation.source_uri,
                        "document_id": item.citation.document_id,
                        "document_content_sha256": (
                            item.citation.document_content_sha256
                        ),
                        "content_sha256": item.citation.content_sha256,
                        "start_char": item.citation.start_char,
                        "end_char": item.citation.end_char,
                        "strategy_id": item.citation.strategy_id,
                    }
                    for item, score in ordered
                ],
            }
        raw_rows.append(
            {
                "query_id": query_id,
                "relevant_ids": sorted(relevant),
                "hybrid_top": candidate_ids[: args.top_k],
                "reranked": reranked_by_depth,
            }
        )

    latency_ids = select_latency_ids(
        eval_ids,
        seed=f"atlasrag-reranking-latency-{args.seed}:",
        count=args.latency_sample,
    )
    hybrid_latency: list[float] = []
    rerank_latency: dict[int, list[float]] = {depth: [] for depth in depths}
    end_to_end_latency: dict[int, list[float]] = {depth: [] for depth in depths}
    latency_samples: list[dict[str, Any]] = []

    for query_id in latency_ids[: min(3, len(latency_ids))]:
        query_text = dataset.queries[query_id]
        candidates = hybrid.search(RetrievalQuery(text=query_text, top_k=max_depth))
        for depth in depths:
            reranker.score(query_text, [item.chunk for item in candidates[:depth]])
            fresh_candidates = hybrid.search(
                RetrievalQuery(text=query_text, top_k=depth)
            )
            reranker.score(
                query_text,
                [item.chunk for item in fresh_candidates],
            )

    for query_id in latency_ids:
        query_text = dataset.queries[query_id]
        synchronize(args.device)
        started = time.perf_counter()
        candidates = hybrid.search(RetrievalQuery(text=query_text, top_k=max_depth))
        synchronize(args.device)
        hybrid_ms = (time.perf_counter() - started) * 1000.0
        hybrid_latency.append(hybrid_ms)
        reranker_sample: dict[str, float] = {}
        end_to_end_sample: dict[str, float] = {}
        for depth in depths:
            synchronize(args.device)
            started = time.perf_counter()
            reranker.score(query_text, [item.chunk for item in candidates[:depth]])
            synchronize(args.device)
            reranker_ms = (time.perf_counter() - started) * 1000.0
            rerank_latency[depth].append(reranker_ms)
            reranker_sample[str(depth)] = reranker_ms

            synchronize(args.device)
            started = time.perf_counter()
            fresh_candidates = hybrid.search(
                RetrievalQuery(text=query_text, top_k=depth)
            )
            reranker.score(
                query_text,
                [item.chunk for item in fresh_candidates],
            )
            synchronize(args.device)
            end_to_end_ms = (time.perf_counter() - started) * 1000.0
            end_to_end_latency[depth].append(end_to_end_ms)
            end_to_end_sample[str(depth)] = end_to_end_ms
        latency_samples.append(
            {
                "query_id": query_id,
                "hybrid_candidate_generation_ms": hybrid_ms,
                "reranker_only_ms": reranker_sample,
                "end_to_end_ms": end_to_end_sample,
            }
        )

    quality = {"hybrid_rrf": mean_metrics(baseline_rows)}
    for depth in depths:
        quality[f"reranked_{depth}"] = mean_metrics(reranked_rows[depth])

    latency = {
        "hybrid_candidate_generation": latency_stats(hybrid_latency),
        "reranker_only": {
            str(depth): latency_stats(rerank_latency[depth]) for depth in depths
        },
        "measured_end_to_end": {
            str(depth): latency_stats(end_to_end_latency[depth]) for depth in depths
        },
        "samples": latency_samples,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.run_id}.rankings.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    summary_csv = args.output_dir / f"{args.run_id}.summary.csv"
    metric_names = [
        f"recall@{args.top_k}",
        f"mrr@{args.top_k}",
        f"ndcg@{args.top_k}",
        f"success@{args.top_k}",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "candidate_depth", "candidate_recall", *metric_names],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "hybrid_rrf",
                "candidate_depth": max_depth,
                "candidate_recall": statistics.fmean(candidate_recall_rows[max_depth]),
                **quality["hybrid_rrf"],
            }
        )
        for depth in depths:
            writer.writerow(
                {
                    "method": "cross_encoder_reranked",
                    "candidate_depth": depth,
                    "candidate_recall": statistics.fmean(candidate_recall_rows[depth]),
                    **quality[f"reranked_{depth}"],
                }
            )

    output = {
        "schema_version": "atlasrag.reranking-benchmark.v1",
        "run_id": args.run_id,
        "dataset": {
            "name": "SciFact",
            "split": "test",
            "query_count": len(eval_ids),
            "corpus_count": len(dataset.corpus),
            "corpus_sha256": sha256(args.dataset_dir / "corpus.jsonl"),
            "queries_sha256": sha256(args.dataset_dir / "queries.jsonl"),
            "qrels_sha256": sha256(args.dataset_dir / "qrels" / "test.tsv"),
            "payload_redistributed": False,
        },
        "package": {
            **package,
            "git_commit": args.atlasrag_git_commit,
            "wheel": args.wheel.name,
            "wheel_sha256": sha256(args.wheel),
        },
        "configuration": {
            "seed": args.seed,
            "top_k": args.top_k,
            "candidate_depths": list(depths),
            "hybrid_component_k": args.hybrid_component_k,
            "rrf_k": args.rrf_k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "embedding_batch_size": args.embedding_batch_size,
            "reranker_batch_size": args.reranker_batch_size,
            "device": args.device,
            "latency_sample": args.latency_sample,
        },
        "models": {
            "dense": {
                "name": DENSE_MODEL_NAME,
                "revision": args.dense_model_revision,
                "license": DENSE_MODEL_LICENSE,
            },
            "reranker": {
                "name": RERANKER_MODEL_NAME,
                "revision": args.reranker_model_revision,
                "license": RERANKER_MODEL_LICENSE,
            },
        },
        "quality": quality,
        "candidate_recall": {
            str(depth): statistics.fmean(candidate_recall_rows[depth])
            for depth in depths
        },
        "latency_ms": latency,
        "setup_seconds": {
            "chunk_build": chunk_build_seconds,
            "index": index_seconds,
            "dense_model_load": embedder.model_load_seconds,
            "embedding_cache_load": embedder.cache_load_seconds,
            "python_vector_conversion": embedder.python_vector_conversion_seconds,
        },
        "environment": environment(),
        "artifacts": {
            "rankings": raw_path.name,
            "rankings_sha256": sha256(raw_path),
            "summary_csv": summary_csv.name,
            "summary_csv_sha256": sha256(summary_csv),
        },
        "limitations": [
            "Single-host local evidence; not a production service-level objective.",
            "Cross-encoder latency uses one RTX 4060 Laptop GPU and excludes model load.",
            "Whole-document SciFact items are a benchmark adapter, not the default chunking strategy.",
            "Candidate recall is an upper bound: reranking cannot recover documents outside the hybrid prefix.",
            "The experiment evaluates one dense model and one cross-encoder revision.",
        ],
    }
    output_path = args.output_dir / f"{args.run_id}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    output["artifact_sha256"] = sha256(output_path)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--dense-model-snapshot", type=Path, required=True)
    parser.add_argument("--dense-model-revision", required=True)
    parser.add_argument("--reranker-model-snapshot", type=Path, required=True)
    parser.add_argument("--reranker-model-revision", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--atlasrag-git-commit", required=True)
    parser.add_argument("--expected-package-version", default="0.3.0.dev0")
    parser.add_argument("--forbid-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-depths", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid-component-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency-sample", type=int, default=25)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
