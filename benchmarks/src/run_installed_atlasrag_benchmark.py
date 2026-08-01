from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
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
    ExactDenseRetriever,
    ReciprocalRankFusionRetriever,
    RetrievalQuery,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
METHODS = ("bm25", "exact_dense", "hybrid_rrf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
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
    name: str
    split: str
    data_dir: Path
    corpus: tuple[dict[str, Any], ...]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    eval_ids: tuple[str, ...]
    exclude_identical_ids: bool


def load_dataset(name: str, data_dir: Path, split: str) -> Dataset:
    corpus = tuple(load_jsonl(data_dir / "corpus.jsonl"))
    queries = {
        str(row["_id"]): str(row.get("text", ""))
        for row in load_jsonl(data_dir / "queries.jsonl")
    }
    qrels = load_qrels(data_dir / "qrels" / f"{split}.tsv")
    return Dataset(
        name=name,
        split=split,
        data_dir=data_dir,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        eval_ids=tuple(sorted(qrels)),
        exclude_identical_ids=name == "arguana",
    )


def row_text(row: dict[str, Any]) -> str:
    return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()


def metrics(
    ranked: Sequence[str], relevant: dict[str, int], top_k: int
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


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one metric row is required")
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def select_latency_ids(query_ids: Sequence[str], *, seed: str, count: int) -> list[str]:
    return sorted(
        query_ids,
        key=lambda query_id: (
            hashlib.sha256((seed + query_id).encode()).hexdigest(),
            query_id,
        ),
    )[: min(count, len(query_ids))]


class FrozenSentenceTransformerEmbedding(EmbeddingModel):
    """Pinned local MiniLM adapter with an explicit corpus embedding cache."""

    def __init__(
        self,
        *,
        snapshot: Path,
        revision: str,
        cache_path: Path,
        rebuild_cache: bool,
        batch_size: int,
        device: str,
    ) -> None:
        self._snapshot = snapshot.resolve()
        self._revision = revision
        self._cache_path = cache_path.resolve()
        self._rebuild_cache = rebuild_cache
        self._batch_size = batch_size
        self._device = device
        self._model: Any | None = None
        self._dimension: int | None = None
        self.model_load_seconds = 0.0
        self.corpus_encode_seconds = 0.0
        self.cache_load_seconds = 0.0
        self.python_vector_conversion_seconds = 0.0
        self.cache_rebuilt = False

    @property
    def model_id(self) -> str:
        return f"{MODEL_NAME}@{self._revision}"

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            self._model = SentenceTransformer(
                str(self._snapshot),
                device=self._device,
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
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self._cache_path.exists() and not self._rebuild_cache:
            started = time.perf_counter()
            matrix = np.load(self._cache_path)
            self.cache_load_seconds = time.perf_counter() - started
        else:
            model = self._load_model()
            started = time.perf_counter()
            matrix = model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self.corpus_encode_seconds = time.perf_counter() - started
            np.save(self._cache_path, matrix)
            self.cache_rebuilt = True
        if matrix.ndim != 2 or matrix.shape != (len(texts), self.dimension):
            raise ValueError(
                "cached corpus embedding shape does not match the frozen corpus"
            )
        started = time.perf_counter()
        vectors = tuple(tuple(float(value) for value in row) for row in matrix)
        self.python_vector_conversion_seconds = time.perf_counter() - started
        return vectors

    def embed_query(self, text: str) -> Vector:
        if not text.strip():
            raise ValueError("query text must not be blank")
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
) -> tuple[tuple[Chunk, ...], dict[str, str], dict[str, str]]:
    chunks: list[Chunk] = []
    external_to_chunk: dict[str, str] = {}
    chunk_to_external: dict[str, str] = {}
    for ordinal, row in enumerate(dataset.corpus):
        external_id = str(row["_id"])
        text = row_text(row)
        source_uri = f"dataset://{dataset.name}/corpus/" + quote(external_id, safe="")
        document = Document.from_text(
            source_uri=source_uri,
            text=text,
            document_id=external_id,
            metadata={
                "dataset.name": dataset.name,
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
        external_to_chunk[external_id] = chunk.chunk_id
        chunk_to_external[chunk.chunk_id] = external_id
    if len(external_to_chunk) != len(chunks):
        raise ValueError("dataset external IDs must be unique")
    return tuple(chunks), external_to_chunk, chunk_to_external


def excluded_chunk_ids_for_query(
    *,
    query_id: str,
    exclude_identical_ids: bool,
    external_to_chunk: dict[str, str],
) -> frozenset[str]:
    if not exclude_identical_ids:
        return frozenset()
    chunk_id = external_to_chunk.get(query_id)
    return frozenset({chunk_id}) if chunk_id is not None else frozenset()


def serialize_result(result: Any, chunk_to_external: dict[str, str]) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "external_id": chunk_to_external[chunk.chunk_id],
        "chunk_id": chunk.chunk_id,
        "rank": result.rank,
        "score": result.score,
        "method": result.method.value,
        "score_kind": result.score_kind.value,
        "document_id": chunk.document_id,
        "document_content_sha256": chunk.document_content_sha256,
        "source_uri": chunk.source_uri,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "content_sha256": chunk.content_sha256,
        "strategy_id": chunk.strategy_id,
        "contributions": [
            {
                "method": item.method.value,
                "rank": item.rank,
                "score": item.score,
                "score_kind": item.score_kind.value,
            }
            for item in result.contributions
        ],
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
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "dependencies": {
            "atlasrag": version("atlasrag"),
            "numpy": version("numpy"),
            "sentence-transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
        },
    }


def assert_installed_package(forbidden_source_root: Path | None) -> dict[str, str]:
    package_file = Path(atlasrag.__file__).resolve()
    distribution_root = Path(
        metadata.distribution("atlasrag").locate_file("")
    ).resolve()
    if forbidden_source_root is not None:
        source_root = forbidden_source_root.resolve()
        if package_file.is_relative_to(source_root):
            raise RuntimeError(
                f"AtlasRAG imported from source tree instead of installed wheel: {package_file}"
            )
    try:
        module_relative_path = str(package_file.relative_to(distribution_root))
    except ValueError as exc:
        raise RuntimeError(
            "installed AtlasRAG module is outside its distribution root"
        ) from exc
    return {
        "version": metadata.version("atlasrag"),
        "import_origin": "installed-wheel-site-packages",
        "module_relative_path": module_relative_path,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    package = assert_installed_package(args.forbid_source_root)
    if package["version"] != args.expected_package_version:
        raise RuntimeError(
            f"installed AtlasRAG version {package['version']} does not match "
            f"expected {args.expected_package_version}"
        )
    if not args.model_snapshot.is_dir():
        raise FileNotFoundError(f"missing local model snapshot: {args.model_snapshot}")

    dataset = load_dataset(args.dataset, args.data_dir, args.split)
    eval_ids = dataset.eval_ids
    if args.query_limit is not None:
        eval_ids = eval_ids[: args.query_limit]
    if not eval_ids:
        raise ValueError("no evaluation query IDs selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    chunks, external_to_chunk, chunk_to_external = build_chunks(dataset)
    chunk_build_seconds = time.perf_counter() - started

    embedder = FrozenSentenceTransformerEmbedding(
        snapshot=args.model_snapshot,
        revision=args.model_revision,
        cache_path=args.embedding_cache,
        rebuild_cache=args.rebuild_embeddings,
        batch_size=args.batch_size,
        device=args.device,
    )
    lexical = BM25Retriever(k1=args.bm25_k1, b=args.bm25_b)
    dense = ExactDenseRetriever(embedder)
    hybrid = ReciprocalRankFusionRetriever(
        lexical,
        dense,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )

    started = time.perf_counter()
    hybrid.index(chunks)
    index_seconds = time.perf_counter() - started

    method_objects = {
        "bm25": lexical,
        "exact_dense": dense,
        "hybrid_rrf": hybrid,
    }
    quality_rows: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    raw_rows: list[dict[str, Any]] = []
    provenance_fields = {
        "document_id",
        "document_content_sha256",
        "source_uri",
        "start_char",
        "end_char",
        "content_sha256",
        "strategy_id",
    }
    provenance_complete = True

    for query_id in eval_ids:
        excluded = excluded_chunk_ids_for_query(
            query_id=query_id,
            exclude_identical_ids=dataset.exclude_identical_ids,
            external_to_chunk=external_to_chunk,
        )
        request = RetrievalQuery(
            text=dataset.queries[query_id],
            top_k=args.candidate_k,
            excluded_chunk_ids=excluded,
        )
        top: dict[str, list[str]] = {}
        detailed: dict[str, list[dict[str, Any]]] = {}
        row_metrics: dict[str, dict[str, float]] = {}
        for method, retriever in method_objects.items():
            serialized = [
                serialize_result(result, chunk_to_external)
                for result in retriever.search(request)
            ]
            ids = [row["external_id"] for row in serialized]
            if dataset.exclude_identical_ids and query_id in ids:
                raise RuntimeError(
                    f"identical query document leaked into {method} results for {query_id}"
                )
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"duplicate result IDs from {method} for {query_id}")
            top[method] = ids[: args.top_k]
            detailed[method] = serialized[: args.top_k]
            row_metrics[method] = metrics(
                ids,
                dataset.qrels[query_id],
                args.top_k,
            )
            quality_rows[method].append(row_metrics[method])
            provenance_complete = provenance_complete and all(
                provenance_fields.issubset(row)
                and all(row[field] not in (None, "") for field in provenance_fields)
                for row in serialized[: args.top_k]
            )
        raw_rows.append(
            {
                "query_id": query_id,
                "relevant_ids": sorted(dataset.qrels[query_id]),
                "excluded_external_ids": [query_id] if excluded else [],
                "top": top,
                "metrics": row_metrics,
                "results": detailed,
            }
        )

    latency_ids = select_latency_ids(
        eval_ids,
        seed="latency-20260731:",
        count=args.latency_sample,
    )
    latency_values: dict[str, list[float]] = {method: [] for method in METHODS}
    warmup_ids = latency_ids[: min(3, len(latency_ids))]
    for query_id in warmup_ids:
        excluded = excluded_chunk_ids_for_query(
            query_id=query_id,
            exclude_identical_ids=dataset.exclude_identical_ids,
            external_to_chunk=external_to_chunk,
        )
        request = RetrievalQuery(
            text=dataset.queries[query_id],
            top_k=args.top_k,
            excluded_chunk_ids=excluded,
        )
        for retriever in method_objects.values():
            retriever.search(request)

    for query_id in latency_ids:
        excluded = excluded_chunk_ids_for_query(
            query_id=query_id,
            exclude_identical_ids=dataset.exclude_identical_ids,
            external_to_chunk=external_to_chunk,
        )
        request = RetrievalQuery(
            text=dataset.queries[query_id],
            top_k=args.top_k,
            excluded_chunk_ids=excluded,
        )
        for method, retriever in method_objects.items():
            started = time.perf_counter()
            retriever.search(request)
            latency_values[method].append((time.perf_counter() - started) * 1000.0)

    raw_path = args.output.with_suffix(".queries.jsonl")
    with raw_path.open("w", encoding="utf-8") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = args.output.with_suffix(".summary.csv")
    quality = {method: mean_metrics(rows) for method, rows in quality_rows.items()}
    metric_fields = [
        f"recall@{args.top_k}",
        f"mrr@{args.top_k}",
        f"ndcg@{args.top_k}",
        f"success@{args.top_k}",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *metric_fields])
        writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, **quality[method]})

    qrels_path = dataset.data_dir / "qrels" / f"{dataset.split}.tsv"
    payload = {
        "schema_version": "atlasrag.installed-package-benchmark.v1",
        "run": {
            "dataset": dataset.name,
            "split": dataset.split,
            "seed": args.seed,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "evaluated_queries": len(eval_ids),
            "identical_query_document_ids_excluded": dataset.exclude_identical_ids,
            "query_ids_present_in_corpus": sum(
                query_id in external_to_chunk for query_id in eval_ids
            ),
            "query_ids_absent_from_corpus": sum(
                query_id not in external_to_chunk for query_id in eval_ids
            ),
            "latency_sample_queries": len(latency_ids),
            "latency_sample_ids": latency_ids,
            "latency_semantics": (
                "installed-package per-query steady-state search including query "
                "embedding for dense and hybrid; model load and index build excluded"
            ),
        },
        "dataset": {
            "corpus_docs": len(dataset.corpus),
            "queries_in_file": len(dataset.queries),
            "evaluated_queries": len(eval_ids),
            "positive_qrels": sum(len(dataset.qrels[item]) for item in eval_ids),
            "corpus_sha256": sha256(dataset.data_dir / "corpus.jsonl"),
            "queries_sha256": sha256(dataset.data_dir / "queries.jsonl"),
            "qrels_sha256": sha256(qrels_path),
        },
        "atlasrag": {
            **package,
            "git_commit": args.atlasrag_git_commit,
            "wheel": args.wheel.name,
            "wheel_sha256": sha256(args.wheel),
            "retrieval_methods": list(METHODS),
            "query_exclusion_applied_before_scoring": True,
        },
        "model": {
            "name": MODEL_NAME,
            "revision": args.model_revision,
            "snapshot": f"local-huggingface-cache/{args.model_revision}",
            "license": MODEL_LICENSE,
            "model_card": MODEL_CARD,
            "device": args.device,
            "load_seconds": embedder.model_load_seconds,
        },
        "build": {
            "chunk_construction_seconds": chunk_build_seconds,
            "combined_index_seconds": index_seconds,
            "corpus_embedding_seconds": embedder.corpus_encode_seconds,
            "embedding_cache_load_seconds": embedder.cache_load_seconds,
            "python_vector_conversion_seconds": (
                embedder.python_vector_conversion_seconds
            ),
            "embedding_cache_rebuilt": embedder.cache_rebuilt,
            "embedding_cache": args.embedding_cache.name,
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "embedding_cache_bytes": args.embedding_cache.stat().st_size,
        },
        "quality": quality,
        "sampled_latency_ms": {
            method: latency_stats(values) for method, values in latency_values.items()
        },
        "contracts": {
            "provenance_complete": provenance_complete,
            "external_id_mapping_complete": len(chunk_to_external) == len(chunks),
            "method_specific_scores_preserved": True,
        },
        "systems": {
            "process_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_measurement_note": (
                "coarse process peak RSS including model, chunks, lexical index, "
                "and Python tuple embeddings"
            ),
        },
        "environment": environment(),
        "artifacts": {
            "raw_queries": raw_path.name,
            "raw_queries_sha256": sha256(raw_path),
            "summary_csv": summary_path.name,
            "summary_csv_sha256": sha256(summary_path),
        },
        "limitations": [
            "Single WSL2 laptop host; no distributed or production-scale claim.",
            "Installed-package latency is not directly comparable to the neutral NumPy harness because the package uses dependency-light Python exact scoring.",
            "The benchmark uses one whole-document chunk per BEIR corpus record.",
            "The ArguAna result is a deterministic 200-query contrast slice, not a full official score.",
            "One model and two public datasets cannot establish universal retrieval method superiority.",
        ],
    }
    if not provenance_complete:
        raise RuntimeError("installed-package result provenance was incomplete")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "arguana"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--atlasrag-git-commit", required=True)
    parser.add_argument("--expected-package-version", default="0.2.0")
    parser.add_argument("--forbid-source-root", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--latency-sample", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
