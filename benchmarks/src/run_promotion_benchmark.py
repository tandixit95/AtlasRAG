"""Run a frozen installed-package depth-10 promotion benchmark."""

from __future__ import annotations

import argparse
import csv
import fcntl
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
    RerankTrace,
    RetrievalMethod,
    RetrievalQuery,
    RetrievalResult,
    ScoreKind,
)

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MODEL_LICENSE = "Apache-2.0"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_LICENSE = "Apache-2.0"
ARGUANA_SELECTION_SEED = "atlasrag-arguana-contrast-20260731:"
METRICS = ("recall@10", "mrr@10", "ndcg@10", "success@10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    task_id: str
    split: str
    corpus: tuple[dict[str, Any], ...]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    eval_ids: tuple[str, ...]
    exclude_identical_ids: bool


def _arguana_ids(qrels: dict[str, dict[str, int]], count: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            qrels,
            key=lambda query_id: (
                hashlib.sha256(
                    (ARGUANA_SELECTION_SEED + query_id).encode()
                ).hexdigest(),
                query_id,
            ),
        )[:count]
    )


def load_dataset(name: str, data_dir: Path) -> Dataset:
    corpus = tuple(load_jsonl(data_dir / "corpus.jsonl"))
    queries = {
        str(row["_id"]): str(row.get("text", ""))
        for row in load_jsonl(data_dir / "queries.jsonl")
    }
    qrels = load_qrels(data_dir / "qrels" / "test.tsv")
    if name == "scifact":
        eval_ids = tuple(sorted(qrels))
        task_id = "scifact-test-300"
        exclude_identical_ids = False
    elif name == "arguana":
        eval_ids = _arguana_ids(qrels, 200)
        task_id = "arguana-contrast-200"
        exclude_identical_ids = True
    else:
        raise ValueError(f"unsupported dataset: {name}")
    if any(query_id not in queries for query_id in eval_ids):
        raise ValueError("qrels reference a missing query")
    return Dataset(
        name=name,
        task_id=task_id,
        split="test",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        eval_ids=eval_ids,
        exclude_identical_ids=exclude_identical_ids,
    )


def row_text(row: dict[str, Any]) -> str:
    return (str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()


def metric_row(
    ranked: Sequence[str], relevant: dict[str, int], *, top_k: int
) -> dict[str, float]:
    top = ranked[:top_k]
    relevant_ids = set(relevant)
    hits = [1 if item_id in relevant_ids else 0 for item_id in top]
    recall = sum(hits) / max(1, len(relevant_ids))
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
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


def dispersion_acceptable(values: dict[str, float]) -> bool:
    p50 = values["p50_ms"]
    if p50 <= 0.0:
        return False
    return values["p95_ms"] <= 2.0 * p50 and values["max_ms"] <= 4.0 * p50


def select_latency_ids(
    query_ids: Sequence[str], *, seed: str, count: int
) -> tuple[str, ...]:
    return tuple(
        sorted(
            query_ids,
            key=lambda query_id: (
                hashlib.sha256((seed + query_id).encode()).hexdigest(),
                query_id,
            ),
        )[: min(count, len(query_ids))]
    )


class FrozenSentenceTransformerEmbedding(EmbeddingModel):
    """Pinned local MiniLM adapter with a checked corpus embedding cache."""

    def __init__(
        self,
        *,
        snapshot: Path,
        revision: str,
        cache_path: Path,
        device: str,
    ) -> None:
        self._snapshot = snapshot.resolve()
        self._revision = revision
        self._cache_path = cache_path.resolve()
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
) -> tuple[tuple[Chunk, ...], dict[str, str], dict[str, str]]:
    chunks: list[Chunk] = []
    external_to_chunk: dict[str, str] = {}
    chunk_to_external: dict[str, str] = {}
    for ordinal, row in enumerate(dataset.corpus):
        external_id = str(row["_id"])
        text = row_text(row)
        document = Document.from_text(
            source_uri=f"dataset://{dataset.name}/corpus/"
            + quote(external_id, safe=""),
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


def load_ratio() -> float:
    logical = os.cpu_count() or 1
    return os.getloadavg()[0] / logical


def idle_samples(*, count: int, interval_seconds: float) -> list[float]:
    values: list[float] = []
    for index in range(count):
        values.append(load_ratio())
        if index + 1 < count:
            time.sleep(interval_seconds)
    return values


def citation_record(item: Any, external_id: str) -> dict[str, Any]:
    citation = item.citation
    return {
        "external_id": external_id,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "document_content_sha256": citation.document_content_sha256,
        "content_sha256": citation.content_sha256,
        "source_uri": citation.source_uri,
        "start_char": citation.start_char,
        "end_char": citation.end_char,
        "strategy_id": citation.strategy_id,
        "reranker_score": item.score,
        "candidate_rank": item.rerank_trace.candidate_rank,
        "candidate_score": item.rerank_trace.candidate_score,
        "candidate_score_kind": item.rerank_trace.candidate_score_kind.value,
        "candidate_method": item.rerank_trace.candidate_method.value,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("frozen protocol requires CUDA for the latency comparison")
    if args.candidate_depth != args.top_k or args.candidate_depth != 10:
        raise ValueError("frozen protocol requires candidate_depth == top_k == 10")
    required_file_paths = (args.embedding_cache, args.wheel)
    if any(not path.is_file() for path in required_file_paths):
        raise FileNotFoundError("required embedding cache or wheel is missing")
    required_directories = (
        args.dataset_dir,
        args.dense_model_snapshot,
        args.reranker_model_snapshot,
    )
    if any(not path.is_dir() for path in required_directories):
        raise FileNotFoundError("required dataset or model snapshot is missing")
    if args.dense_model_snapshot.resolve().name != args.dense_model_revision:
        raise ValueError("dense model snapshot path does not match frozen revision")
    if args.reranker_model_snapshot.resolve().name != args.reranker_model_revision:
        raise ValueError("reranker snapshot path does not match frozen revision")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = args.lock_file.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "another promotion benchmark holds the exclusive lock"
        ) from exc

    preflight = idle_samples(count=3, interval_seconds=1.0)
    if max(preflight) > args.max_load_ratio:
        raise RuntimeError(
            f"host preflight load ratio {max(preflight):.3f} exceeds "
            f"{args.max_load_ratio:.3f}"
        )

    package = assert_installed_package(
        expected_version=args.expected_package_version,
        forbidden_source_root=args.forbid_source_root,
    )
    dataset = load_dataset(args.dataset, args.dataset_dir)
    if args.task_id != dataset.task_id:
        raise ValueError(f"task ID {args.task_id} does not match {dataset.task_id}")
    chunks, external_to_chunk, chunk_to_external = build_chunks(dataset)

    embedder = FrozenSentenceTransformerEmbedding(
        snapshot=args.dense_model_snapshot,
        revision=args.dense_model_revision,
        cache_path=args.embedding_cache,
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

    baseline_metric_rows: list[dict[str, float]] = []
    candidate_metric_rows: list[dict[str, float]] = []
    raw_rows: list[dict[str, Any]] = []
    identical_present = 0

    for query_id in dataset.eval_ids:
        query_text = dataset.queries[query_id]
        relevant = dataset.qrels[query_id]
        excluded = frozenset()
        identical_chunk = external_to_chunk.get(query_id)
        if dataset.exclude_identical_ids and identical_chunk is not None:
            identical_present += 1
            excluded = frozenset({identical_chunk})
        query = RetrievalQuery(
            text=query_text,
            top_k=args.candidate_depth,
            excluded_chunk_ids=excluded,
        )
        candidates = hybrid.search(query)
        candidate_ids = [chunk_to_external[item.chunk.chunk_id] for item in candidates]
        if query_id in candidate_ids and dataset.exclude_identical_ids:
            raise RuntimeError(f"identical query document leaked for {query_id}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(f"duplicate candidate for {query_id}")
        scores = tuple(
            float(value)
            for value in reranker.score(query_text, [item.chunk for item in candidates])
        )
        if len(scores) != len(candidates):
            raise RuntimeError("reranker returned an invalid score count")
        ordered = list(zip(candidates, scores, strict=True))
        ordered.sort(key=lambda item: (-item[1], item[0].rank, item[0].chunk.chunk_id))
        reranked_ids = [
            chunk_to_external[item.chunk.chunk_id]
            for item, _score in ordered[: args.top_k]
        ]
        baseline_metrics = metric_row(candidate_ids, relevant, top_k=args.top_k)
        candidate_metrics = metric_row(reranked_ids, relevant, top_k=args.top_k)
        baseline_metric_rows.append(baseline_metrics)
        candidate_metric_rows.append(candidate_metrics)

        reranked_top = []
        for final_rank, (candidate, score) in enumerate(ordered[: args.top_k], start=1):
            reranked_result = RetrievalResult(
                chunk=candidate.chunk,
                score=score,
                rank=final_rank,
                method=RetrievalMethod.RERANKED,
                score_kind=ScoreKind.RERANKER_RELEVANCE,
                contributions=candidate.contributions,
                rerank_trace=RerankTrace(
                    model_id=reranker.model_id,
                    candidate_method=candidate.method,
                    candidate_score_kind=candidate.score_kind,
                    candidate_score=candidate.score,
                    candidate_rank=candidate.rank,
                    candidate_contributions=candidate.contributions,
                ),
            )
            reranked_top.append(
                citation_record(
                    reranked_result,
                    chunk_to_external[candidate.chunk.chunk_id],
                )
            )
        raw_rows.append(
            {
                "query_id": query_id,
                "relevant_ids": sorted(relevant),
                "excluded_external_ids": [query_id] if excluded else [],
                "baseline_top": candidate_ids[: args.top_k],
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "reranked_top": reranked_top,
            }
        )

    latency_ids = select_latency_ids(
        dataset.eval_ids,
        seed=f"atlasrag-promotion-latency-{dataset.task_id}-{args.seed}:",
        count=args.latency_sample,
    )
    for query_id in latency_ids[: min(3, len(latency_ids))]:
        excluded = frozenset()
        identical_chunk = external_to_chunk.get(query_id)
        if dataset.exclude_identical_ids and identical_chunk is not None:
            excluded = frozenset({identical_chunk})
        query_text = dataset.queries[query_id]
        candidates = hybrid.search(
            RetrievalQuery(
                text=query_text,
                top_k=args.candidate_depth,
                excluded_chunk_ids=excluded,
            )
        )
        reranker.score(query_text, [item.chunk for item in candidates])

    hybrid_samples: list[float] = []
    reranker_samples: list[float] = []
    end_to_end_samples: list[float] = []
    load_samples: list[float] = []
    raw_latency: list[dict[str, Any]] = []
    for query_id in latency_ids:
        ratio = load_ratio()
        load_samples.append(ratio)
        excluded = frozenset()
        identical_chunk = external_to_chunk.get(query_id)
        if dataset.exclude_identical_ids and identical_chunk is not None:
            excluded = frozenset({identical_chunk})
        query_text = dataset.queries[query_id]
        request = RetrievalQuery(
            text=query_text,
            top_k=args.candidate_depth,
            excluded_chunk_ids=excluded,
        )
        synchronize(args.device)
        started = time.perf_counter()
        candidates = hybrid.search(request)
        synchronize(args.device)
        hybrid_ms = (time.perf_counter() - started) * 1000.0

        synchronize(args.device)
        started = time.perf_counter()
        reranker.score(query_text, [item.chunk for item in candidates])
        synchronize(args.device)
        reranker_ms = (time.perf_counter() - started) * 1000.0

        synchronize(args.device)
        started = time.perf_counter()
        fresh = hybrid.search(request)
        reranker.score(query_text, [item.chunk for item in fresh])
        synchronize(args.device)
        end_to_end_ms = (time.perf_counter() - started) * 1000.0

        hybrid_samples.append(hybrid_ms)
        reranker_samples.append(reranker_ms)
        end_to_end_samples.append(end_to_end_ms)
        raw_latency.append(
            {
                "query_id": query_id,
                "load_ratio_before_sample": ratio,
                "hybrid_candidate_generation_ms": hybrid_ms,
                "reranker_only_ms": reranker_ms,
                "end_to_end_ms": end_to_end_ms,
            }
        )

    baseline_quality = mean_metrics(baseline_metric_rows)
    candidate_quality = mean_metrics(candidate_metric_rows)
    hybrid_latency = latency_stats(hybrid_samples)
    reranker_latency = latency_stats(reranker_samples)
    end_to_end_latency = latency_stats(end_to_end_samples)
    host_passed = max([*preflight, *load_samples], default=0.0) <= args.max_load_ratio

    raw_path = args.output_dir / f"{args.run_id}.rankings.jsonl"
    with raw_path.open("w", encoding="ascii", newline="\n") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    summary_path = args.output_dir / f"{args.run_id}.summary.csv"
    with summary_path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "candidate_depth", *METRICS],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "hybrid_rrf",
                "candidate_depth": args.candidate_depth,
                **baseline_quality,
            }
        )
        writer.writerow(
            {
                "method": "cross_encoder_reranked",
                "candidate_depth": args.candidate_depth,
                **candidate_quality,
            }
        )

    output = {
        "schema_version": "atlasrag.promotion-benchmark.v1",
        "protocol_id": args.protocol_id,
        "run_id": args.run_id,
        "task_id": dataset.task_id,
        "dataset": {
            "name": dataset.name,
            "split": dataset.split,
            "query_count": len(dataset.eval_ids),
            "corpus_count": len(dataset.corpus),
            "corpus_sha256": sha256(args.dataset_dir / "corpus.jsonl"),
            "queries_sha256": sha256(args.dataset_dir / "queries.jsonl"),
            "qrels_sha256": sha256(args.dataset_dir / "qrels" / "test.tsv"),
            "selection_rule": (
                "full sorted positive-qrel test set"
                if dataset.name == "scifact"
                else "SHA256(atlasrag-arguana-contrast-20260731: + query_id), then query_id; take 200"
            ),
            "identical_query_document_ids_excluded": dataset.exclude_identical_ids,
            "query_ids_present_in_corpus": identical_present,
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
            "candidate_depth": args.candidate_depth,
            "hybrid_component_k": args.hybrid_component_k,
            "rrf_k": args.rrf_k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
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
        "quality": {
            "hybrid_rrf": baseline_quality,
            "cross_encoder_reranked": candidate_quality,
        },
        "latency_ms": {
            "hybrid_candidate_generation": hybrid_latency,
            "reranker_only": reranker_latency,
            "measured_end_to_end": end_to_end_latency,
            "samples": raw_latency,
            "dispersion_acceptable": dispersion_acceptable(reranker_latency),
            "dispersion_rule": "reranker p95 <= 2x p50 and max <= 4x p50",
        },
        "host_control": {
            "exclusive_benchmark_lock": True,
            "offline_model_loading": True,
            "one_benchmark_process": True,
            "max_load_ratio": args.max_load_ratio,
            "preflight_load_ratio_samples": preflight,
            "latency_load_ratio_samples": load_samples,
            "passed": host_passed,
            "scope": "controlled local benchmark process, not production isolation",
        },
        "setup_seconds": {
            "index": index_seconds,
            "dense_model_load": embedder.model_load_seconds,
            "embedding_cache_load": embedder.cache_load_seconds,
            "python_vector_conversion": embedder.python_vector_conversion_seconds,
        },
        "environment": environment(),
        "artifacts": {
            "rankings": raw_path.name,
            "rankings_sha256": sha256(raw_path),
            "summary_csv": summary_path.name,
            "summary_csv_sha256": sha256(summary_path),
        },
        "limitations": [
            "Controlled single-host component evidence; not a production service-level objective.",
            "Whole BEIR documents are benchmark retrieval units, not the default chunking strategy.",
            "The ArguAna task is a deterministic 200-query contrast slice, not a full official score.",
            "The experiment evaluates one dense model and one cross-encoder revision.",
            "No query text, corpus text, model weights, or embedding cache is redistributed.",
        ],
    }
    output_path = args.output_dir / f"{args.run_id}.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset", choices=["scifact", "arguana"], required=True)
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
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hybrid-component-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency-sample", type=int, default=25)
    parser.add_argument("--max-load-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
