from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    ExactDenseRetriever,
    PermissionPolicy,
    ReciprocalRankFusionRetriever,
    RetrievalQuery,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class StableTokenEmbedder(EmbeddingModel):
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "synthetic/stable-token-hash-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> Vector:
        values = [0.0] * self._dimension
        for token in TOKEN_RE.findall(text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[idx] += sign
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            values[0] = 1.0
            norm = 1.0
        return tuple(v / norm for v in values)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


def to_chunk(row: dict, ordinal: int) -> Chunk:
    groups = frozenset(row["groups"])
    policy = PermissionPolicy(
        tenant_id=None if groups == {"all"} else row["tenant_id"],
        allowed_groups=frozenset() if groups == {"all"} else groups,
    )
    doc = Document.from_text(
        source_uri=row["source_uri"],
        text=row["text"],
        document_id=row["chunk_id"],
        metadata=policy.to_metadata()
        | {
            "synthetic.index_version": str(row["index_version"]),
            "synthetic.shard_id": str(row["shard_id"]),
        },
    )
    return Chunk.from_document_span(
        document=doc,
        start_char=0,
        end_char=len(doc.text),
        ordinal=ordinal,
        strategy_id="synthetic-whole-document-v1",
    )


def serialize(results):
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "rank": r.rank,
            "score": r.score,
            "method": r.method.value,
            "score_kind": r.score_kind.value,
            "document_id": r.chunk.document_id,
            "document_content_sha256": r.chunk.document_content_sha256,
            "source_uri": r.chunk.source_uri,
            "start_char": r.chunk.start_char,
            "end_char": r.chunk.end_char,
            "content_sha256": r.chunk.content_sha256,
            "strategy_id": r.chunk.strategy_id,
            "contributions": [
                {
                    "method": c.method.value,
                    "rank": c.rank,
                    "score": c.score,
                    "score_kind": c.score_kind.value,
                }
                for c in r.contributions
            ],
        }
        for r in results
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    fixture = json.loads(args.dataset.read_text())
    chunks = tuple(to_chunk(row, i) for i, row in enumerate(fixture["documents"]))
    embedder = StableTokenEmbedder()
    lexical = BM25Retriever(k1=1.5, b=0.75)
    dense = ExactDenseRetriever(embedder)
    hybrid = ReciprocalRankFusionRetriever(lexical, dense, rrf_k=60, candidate_k=100)
    hybrid.index(chunks)

    methods = {"bm25": lexical, "exact_dense": dense, "hybrid_rrf": hybrid}
    checks = []
    provenance_fields = {
        "document_id",
        "document_content_sha256",
        "source_uri",
        "start_char",
        "end_char",
        "content_sha256",
        "strategy_id",
    }

    for scenario in fixture["scenarios"]:
        if scenario["scenario_id"] not in {
            "tenant_isolation",
            "group_isolation",
            "provenance_complete",
        }:
            continue
        groups = frozenset(g for g in scenario["groups"] if g != "all")
        principal = AccessPrincipal(tenant_id=scenario["tenant_id"], groups=groups)
        request = RetrievalQuery(text=scenario["query"], top_k=3, principal=principal)
        for method_name, retriever in methods.items():
            raw = serialize(retriever.search(request))
            returned = {r["chunk_id"] for r in raw}
            forbidden = set(scenario["expect"]["forbidden_ids"])
            checks.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "method": method_name,
                    "returned_ids": sorted(returned),
                    "unauthorized_ids": sorted(returned & forbidden),
                    "provenance_complete": all(
                        provenance_fields.issubset(r)
                        and all(r[f] not in (None, "") for f in provenance_fields)
                        for r in raw
                    ),
                    "results": raw,
                }
            )

    # Security invariant: an unauthorized high-overlap chunk must not perturb visible BM25 score/rank.
    alpha_public = next(c for c in chunks if c.document_id == "a-public-01")
    beta_public = next(c for c in chunks if c.document_id == "b-public-01")
    protected_beta_doc = Document.from_text(
        source_uri=beta_public.source_uri + "/protected-copy",
        text=beta_public.text,
        document_id="b-public-protected-copy",
        metadata=PermissionPolicy(tenant_id="beta").to_metadata(),
    )
    protected_beta = Chunk.from_document_span(
        document=protected_beta_doc,
        start_char=0,
        end_char=len(protected_beta_doc.text),
        ordinal=0,
        strategy_id="synthetic-whole-document-v1",
    )
    principal = AccessPrincipal(tenant_id="alpha")
    request = RetrievalQuery(
        text="battery inspection interval cycles", top_k=3, principal=principal
    )
    base = BM25Retriever()
    base.index((alpha_public,))
    perturbed = BM25Retriever()
    perturbed.index((alpha_public, protected_beta))
    base_result = serialize(base.search(request))
    perturbed_result = serialize(perturbed.search(request))
    bm25_invariant = base_result == perturbed_result

    unauthorized_count = sum(len(c["unauthorized_ids"]) for c in checks)
    provenance_complete = all(c["provenance_complete"] for c in checks)
    deterministic_a = json.dumps(checks, sort_keys=True)
    # Repeat all searches without reindexing.
    repeat = []
    for scenario in fixture["scenarios"]:
        if scenario["scenario_id"] not in {
            "tenant_isolation",
            "group_isolation",
            "provenance_complete",
        }:
            continue
        groups = frozenset(g for g in scenario["groups"] if g != "all")
        request = RetrievalQuery(
            text=scenario["query"],
            top_k=3,
            principal=AccessPrincipal(tenant_id=scenario["tenant_id"], groups=groups),
        )
        for method_name, retriever in methods.items():
            raw = serialize(retriever.search(request))
            returned = {r["chunk_id"] for r in raw}
            forbidden = set(scenario["expect"]["forbidden_ids"])
            repeat.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "method": method_name,
                    "returned_ids": sorted(returned),
                    "unauthorized_ids": sorted(returned & forbidden),
                    "provenance_complete": all(
                        provenance_fields.issubset(r)
                        and all(r[f] not in (None, "") for f in provenance_fields)
                        for r in raw
                    ),
                    "results": raw,
                }
            )
    deterministic = deterministic_a == json.dumps(repeat, sort_keys=True)

    payload = {
        "schema_version": "atlasrag.adapter-smoke.v1",
        "atlasrag_commit": "5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1",
        "fixture": str(args.dataset),
        "methods": list(methods),
        "metrics": {
            "authorization_checks": len(checks),
            "unauthorized_return_count": unauthorized_count,
            "provenance_completeness": 1.0 if provenance_complete else 0.0,
            "bm25_unauthorized_score_rank_invariant": bm25_invariant,
            "deterministic_rerun": deterministic,
        },
        "gates": {
            "passed": unauthorized_count == 0
            and provenance_complete
            and bm25_invariant
            and deterministic,
        },
        "checks": checks,
        "limitations": [
            "This is an integrated API smoke test over a small synthetic fixture, not the public quality benchmark.",
            "The deterministic token-hash embedder validates retrieval contracts and filtering, not embedding quality.",
            "Missing-shard, stale-index, and unsupported-query behavior remain neutral adapter-level contracts because AtlasRAG 0.2.0 has no distributed index coordinator.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not payload["gates"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
