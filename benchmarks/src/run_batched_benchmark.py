from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import math
import os
import pickle
import platform
import re
import resource
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                out[str(row["query-id"])][str(row["corpus-id"])] = score
    return dict(out)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(m.group(0) for m in TOKEN_RE.finditer(text.casefold()))


@dataclass(frozen=True)
class Dataset:
    name: str
    data_dir: Path
    split: str
    corpus: list[dict[str, Any]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    eval_ids: list[str]
    exclude_identical_ids: bool


def load_dataset(name: str, data_dir: Path, split: str) -> Dataset:
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = {
        str(q["_id"]): str(q.get("text", ""))
        for q in load_jsonl(data_dir / "queries.jsonl")
    }
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    qrels = load_qrels(qrels_path)
    return Dataset(
        name=name,
        data_dir=data_dir,
        split=split,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        eval_ids=sorted(qrels),
        exclude_identical_ids=(name == "arguana"),
    )


class BM25Index:
    def __init__(
        self,
        doc_ids: list[str],
        doc_lengths: np.ndarray,
        postings: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.doc_ids = doc_ids
        self.doc_lengths = doc_lengths.astype(np.float64, copy=False)
        self.postings = postings
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(doc_ids)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.n_docs else 0.0

    @classmethod
    def build(
        cls, doc_ids: list[str], texts: list[str], *, k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        postings_lists: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(texts), dtype=np.int32)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings_lists[term].append((i, tf))
        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, rows in postings_lists.items():
            postings[term] = (
                np.fromiter((r[0] for r in rows), dtype=np.int32, count=len(rows)),
                np.fromiter((r[1] for r in rows), dtype=np.float32, count=len(rows)),
            )
        return cls(doc_ids, lengths, postings, k1=k1, b=b)

    def search(
        self, query: str, candidate_k: int, exclude_id: str | None = None
    ) -> list[str]:
        query_terms = Counter(tokenize(query))
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term, qtf in query_terms.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, tfs = posting
            df = len(docs)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            length_ratio = self.doc_lengths[docs] / self.avgdl if self.avgdl else 0.0
            denom = tfs + self.k1 * (1.0 - self.b + self.b * length_ratio)
            scores[docs] += qtf * idf * (tfs * (self.k1 + 1.0) / denom)
        positive = np.flatnonzero(scores > 0.0)
        if positive.size == 0:
            return []
        take = min(candidate_k + (1 if exclude_id else 0), positive.size)
        if take < positive.size:
            local = np.argpartition(-scores[positive], take - 1)[:take]
            selected = positive[local]
        else:
            selected = positive
        ordered = sorted(
            (int(i) for i in selected),
            key=lambda i: (-float(scores[i]), self.doc_ids[i]),
        )
        return [self.doc_ids[i] for i in ordered if self.doc_ids[i] != exclude_id][
            :candidate_k
        ]


def load_or_build_bm25(
    dataset: Dataset, path: Path, rebuild: bool
) -> tuple[BM25Index, float]:
    if path.exists() and not rebuild:
        with path.open("rb") as f:
            state = pickle.load(f)
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported BM25 cache format; rebuild the cache")
        return BM25Index(
            list(state["doc_ids"]),
            np.asarray(state["doc_lengths"]),
            dict(state["postings"]),
            k1=float(state["k1"]),
            b=float(state["b"]),
        ), 0.0
    texts = [
        (str(d.get("title", "")) + "\n" + str(d.get("text", ""))).strip()
        for d in dataset.corpus
    ]
    doc_ids = [str(d["_id"]) for d in dataset.corpus]
    t0 = time.perf_counter()
    index = BM25Index.build(doc_ids, texts)
    elapsed = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "doc_ids": index.doc_ids,
        "doc_lengths": index.doc_lengths,
        "postings": index.postings,
        "k1": index.k1,
        "b": index.b,
    }
    with path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return index, elapsed


def resolve_model_revision() -> str | None:
    base = (
        Path.home()
        / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    )
    if not base.exists():
        return None
    revisions = sorted(p.name for p in base.iterdir() if p.is_dir())
    return revisions[-1] if revisions else None


def load_or_build_embeddings(
    dataset: Dataset,
    model: SentenceTransformer,
    path: Path,
    rebuild: bool,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    if path.exists() and not rebuild:
        return np.load(path, mmap_mode="r"), 0.0
    texts = [
        (str(d.get("title", "")) + "\n" + str(d.get("text", ""))).strip()
        for d in dataset.corpus
    ]
    t0 = time.perf_counter()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embs)
    return embs, elapsed


def load_or_build_hnsw(
    embeddings: np.ndarray, path: Path, rebuild: bool, seed: int, ef: int
) -> tuple[hnswlib.Index, float]:
    idx = hnswlib.Index(space="cosine", dim=int(embeddings.shape[1]))
    if path.exists() and not rebuild:
        idx.load_index(str(path))
        idx.set_ef(ef)
        return idx, 0.0
    t0 = time.perf_counter()
    idx.init_index(
        max_elements=len(embeddings), ef_construction=160, M=16, random_seed=seed
    )
    idx.add_items(np.asarray(embeddings), np.arange(len(embeddings), dtype=np.int64))
    idx.set_ef(ef)
    idx.save_index(str(path))
    return idx, time.perf_counter() - t0


def ranked_from_scores(
    scores: np.ndarray, doc_ids: list[str], candidate_k: int, exclude_id: str | None
) -> list[str]:
    take = min(candidate_k + (1 if exclude_id else 0), len(scores))
    if take < len(scores):
        selected = np.argpartition(-scores, take - 1)[:take]
    else:
        selected = np.arange(len(scores))
    ordered = sorted(
        (int(i) for i in selected), key=lambda i: (-float(scores[i]), doc_ids[i])
    )
    return [doc_ids[i] for i in ordered if doc_ids[i] != exclude_id][:candidate_k]


def rrf(a: list[str], b: list[str], rrf_k: int, top_k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for seq in (a, b):
        for rank, doc_id in enumerate(seq, 1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    ]


def metrics(ranked: list[str], relevant: dict[str, int], k: int) -> dict[str, float]:
    top = ranked[:k]
    rel = set(relevant)
    hits = [1 if d in rel else 0 for d in top]
    recall = sum(hits) / max(1, len(rel))
    mrr = next((1.0 / r for r, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(r + 1) for r, hit in enumerate(hits, 1))
    ideal = min(k, len(rel))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return {
        f"recall@{k}": recall,
        f"mrr@{k}": mrr,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"success@{k}": float(any(hits)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}


def env() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
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
            "numpy": version("numpy"),
            "sentence-transformers": version("sentence-transformers"),
            "transformers": version("transformers"),
            "hnswlib": version("hnswlib"),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset, args.data_dir, args.split)
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc_ids = [str(d["_id"]) for d in dataset.corpus]
    bm25_path = args.workdir / "bm25.pkl"
    emb_path = args.workdir / "corpus_embeddings.npy"
    hnsw_path = args.workdir / "dense.hnsw"

    bm25, bm25_build_s = load_or_build_bm25(dataset, bm25_path, args.rebuild)
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    model_load_s = time.perf_counter() - t0
    corpus_embs, embedding_build_s = load_or_build_embeddings(
        dataset, model, emb_path, args.rebuild, args.batch_size
    )
    hnsw, hnsw_build_s = load_or_build_hnsw(
        corpus_embs, hnsw_path, args.rebuild, args.seed, args.ann_ef
    )

    query_texts = [dataset.queries[qid] for qid in dataset.eval_ids]
    t0 = time.perf_counter()
    query_embs = model.encode(
        query_texts,
        batch_size=args.query_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    batch_query_encode_s = time.perf_counter() - t0
    exact_matrix = query_embs @ np.asarray(corpus_embs).T
    ann_labels, _ = hnsw.knn_query(
        query_embs, k=min(args.candidate_k + 1, len(doc_ids))
    )

    quality = {m: [] for m in ("bm25", "exact_dense", "ann_hnsw", "hybrid_rrf")}
    raw = []
    ann_overlap = []
    for row_idx, qid in enumerate(dataset.eval_ids):
        exclude = qid if dataset.exclude_identical_ids else None
        bm = bm25.search(dataset.queries[qid], args.candidate_k, exclude)
        exact = ranked_from_scores(
            exact_matrix[row_idx], doc_ids, args.candidate_k, exclude
        )
        ann = [
            doc_ids[int(i)] for i in ann_labels[row_idx] if doc_ids[int(i)] != exclude
        ][: args.candidate_k]
        hybrid = rrf(bm, exact, args.rrf_k, args.candidate_k)
        ranked = {
            "bm25": bm,
            "exact_dense": exact,
            "ann_hnsw": ann,
            "hybrid_rrf": hybrid,
        }
        row_metrics = {
            m: metrics(ids, dataset.qrels[qid], args.top_k) for m, ids in ranked.items()
        }
        for m, rows in quality.items():
            rows.append(row_metrics[m])
        ann_overlap.append(
            len(set(ann[: args.top_k]) & set(exact[: args.top_k])) / args.top_k
        )
        raw.append(
            {
                "query_id": qid,
                "relevant_ids": sorted(dataset.qrels[qid]),
                "top": {m: ids[: args.top_k] for m, ids in ranked.items()},
                "metrics": row_metrics,
            }
        )

    sample_ids = sorted(
        dataset.eval_ids,
        key=lambda q: (
            hashlib.sha256(("latency-20260731:" + q).encode()).hexdigest(),
            q,
        ),
    )[: min(args.latency_sample, len(dataset.eval_ids))]
    lat = {
        k: []
        for k in (
            "query_encode",
            "bm25_search",
            "exact_search",
            "ann_search",
            "rrf_fusion",
            "bm25_end_to_end",
            "exact_end_to_end",
            "ann_end_to_end",
            "hybrid_sequential",
            "hybrid_concurrent",
        )
    }
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        for qid in sample_ids:
            text = dataset.queries[qid]
            exclude = qid if dataset.exclude_identical_ids else None
            t0 = time.perf_counter()
            qvec = model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
            encode_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            bm = bm25.search(text, args.candidate_k, exclude)
            bm_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            exact = ranked_from_scores(
                np.asarray(corpus_embs) @ qvec, doc_ids, args.candidate_k, exclude
            )
            exact_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            labels, _ = hnsw.knn_query(
                qvec.reshape(1, -1), k=min(args.candidate_k + 1, len(doc_ids))
            )
            ann = [doc_ids[int(i)] for i in labels[0] if doc_ids[int(i)] != exclude][
                : args.candidate_k
            ]
            ann_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            hybrid = rrf(bm, exact, args.rrf_k, args.candidate_k)
            fusion_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            f1 = executor.submit(bm25.search, text, args.candidate_k, exclude)
            f2 = executor.submit(
                lambda qvec=qvec, exclude=exclude: ranked_from_scores(
                    np.asarray(corpus_embs) @ qvec, doc_ids, args.candidate_k, exclude
                )
            )
            cbm = f1.result()
            cex = f2.result()
            chy = rrf(cbm, cex, args.rrf_k, args.candidate_k)
            concurrent_ms = (time.perf_counter() - t0) * 1000 + encode_ms
            if chy != hybrid:
                raise RuntimeError(f"concurrent rank mismatch for {qid}")
            lat["query_encode"].append(encode_ms)
            lat["bm25_search"].append(bm_ms)
            lat["exact_search"].append(exact_ms)
            lat["ann_search"].append(ann_ms)
            lat["rrf_fusion"].append(fusion_ms)
            lat["bm25_end_to_end"].append(bm_ms)
            lat["exact_end_to_end"].append(encode_ms + exact_ms)
            lat["ann_end_to_end"].append(encode_ms + ann_ms)
            lat["hybrid_sequential"].append(encode_ms + bm_ms + exact_ms + fusion_ms)
            lat["hybrid_concurrent"].append(concurrent_ms)
    finally:
        executor.shutdown(wait=True)

    raw_path = args.output.with_suffix(".queries.jsonl")
    with raw_path.open("w", encoding="utf-8") as f:
        for row in raw:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method",
            f"recall@{args.top_k}",
            f"mrr@{args.top_k}",
            f"ndcg@{args.top_k}",
            f"success@{args.top_k}",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, rows in quality.items():
            w.writerow({"method": method, **mean_metrics(rows)})

    qrels_path = dataset.data_dir / "qrels" / f"{args.split}.tsv"
    payload = {
        "schema_version": "atlasrag.batched-retrieval-benchmark.v1",
        "run": {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "ann_ef": args.ann_ef,
            "evaluated_queries": len(dataset.eval_ids),
            "identical_query_document_ids_excluded": dataset.exclude_identical_ids,
            "latency_sample_queries": len(sample_ids),
            "latency_sample_ids": sample_ids,
            "quality_query_encoding": "batched offline evaluation pass",
            "latency_semantics": "per-query sampled steady-state timings; offline index build/model load excluded",
        },
        "dataset": {
            "corpus_docs": len(dataset.corpus),
            "queries_in_file": len(dataset.queries),
            "evaluated_queries": len(dataset.eval_ids),
            "positive_qrels": sum(len(v) for v in dataset.qrels.values()),
            "corpus_sha256": sha256(dataset.data_dir / "corpus.jsonl"),
            "queries_sha256": sha256(dataset.data_dir / "queries.jsonl"),
            "qrels_sha256": sha256(qrels_path),
        },
        "model": {
            "name": MODEL_NAME,
            "license": MODEL_LICENSE,
            "model_card": MODEL_CARD,
            "cached_revision": resolve_model_revision(),
            "load_seconds": model_load_s,
        },
        "build": {
            "bm25_seconds": bm25_build_s,
            "corpus_embedding_seconds": embedding_build_s,
            "hnsw_seconds": hnsw_build_s,
            "batch_query_encoding_seconds": batch_query_encode_s,
            "artifact_bytes": {
                "bm25": bm25_path.stat().st_size,
                "embeddings": emb_path.stat().st_size,
                "hnsw": hnsw_path.stat().st_size,
                "total": bm25_path.stat().st_size
                + emb_path.stat().st_size
                + hnsw_path.stat().st_size,
            },
        },
        "quality": {m: mean_metrics(rows) for m, rows in quality.items()},
        "sampled_latency_ms": {k: latency_stats(v) for k, v in lat.items()},
        "ann_vs_exact": {
            f"mean_top{args.top_k}_overlap": statistics.fmean(ann_overlap),
            "quality_reference": "exact_dense",
            "note": "ANN is an approximation tradeoff only; exact dense remains the correctness reference.",
        },
        "systems": {
            "process_peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "memory_measurement_note": "coarse process peak RSS including model and all in-process artifacts",
        },
        "environment": env(),
        "artifacts": {
            "raw_queries": raw_path.name,
            "raw_queries_sha256": sha256(raw_path),
            "summary_csv": csv_path.name,
            "summary_csv_sha256": sha256(csv_path),
        },
        "limitations": [
            "Single WSL2 laptop host; no distributed or production-scale claim.",
            "Latency is measured on a deterministic query sample while quality uses the full frozen evaluation slice.",
            "Exact dense is exhaustive NumPy cosine search and is a correctness baseline, not a scalable serving design.",
            "One model and two public datasets cannot establish universal retrieval method superiority.",
        ],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["scifact", "arguana"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--ann-ef", type=int, default=100)
    p.add_argument("--latency-sample", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--query-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260731)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
