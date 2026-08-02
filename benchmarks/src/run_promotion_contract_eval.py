"""Exercise default-promotion safety contracts through the installed package API."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

import atlasrag
from atlasrag.embeddings.base import EmbeddingModel, Vector
from atlasrag.ingestion import FixedCharacterChunker
from atlasrag.models import Chunk, Document
from atlasrag.retrieval import (
    AccessPrincipal,
    BM25Retriever,
    ExactDenseRetriever,
    PermissionPolicy,
    ReciprocalRankFusionRetriever,
    RerankedRetriever,
    Reranker,
    RetrievalQuery,
)
from atlasrag.retrieval.access import GROUPS_METADATA_KEY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ContractEmbedding(EmbeddingModel):
    @property
    def model_id(self) -> str:
        return "promotion-contract-embedding-v1"

    @property
    def dimension(self) -> int:
        return 3

    def _embed(self, text: str) -> Vector:
        lowered = text.casefold()
        return (
            float("mars" in lowered),
            float("private" in lowered or "restricted" in lowered),
            1.0,
        )

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class RecordingReranker(Reranker):
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.scored_source_uris: list[str] = []

    @property
    def model_id(self) -> str:
        return "promotion-contract-reranker-v1"

    def score(self, query: str, chunks: Sequence[Chunk]) -> tuple[float, ...]:
        self.scored_source_uris.extend(chunk.source_uri for chunk in chunks)
        return tuple(self._scores[chunk.source_uri] for chunk in chunks)


def chunk(
    name: str,
    text: str,
    *,
    policy: PermissionPolicy | None = None,
    metadata: dict[str, str] | None = None,
) -> Chunk:
    combined = dict(metadata or {})
    if policy is not None:
        combined.update(policy.to_metadata())
    document = Document.from_text(
        source_uri=f"memory://promotion/{name}",
        text=text,
        metadata=combined,
    )
    return FixedCharacterChunker(chunk_size=1000).chunk(document)[0]


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


def build_retriever(chunks: Sequence[Chunk], reranker: RecordingReranker):
    hybrid = ReciprocalRankFusionRetriever(
        BM25Retriever(),
        ExactDenseRetriever(ContractEmbedding()),
        candidate_k=10,
    )
    hybrid.index(chunks)
    return RerankedRetriever(hybrid, reranker, candidate_k=10)


def citation_complete(result) -> bool:
    citation = result.citation
    return bool(
        citation.chunk_id
        and citation.document_id
        and citation.document_content_sha256
        and citation.content_sha256
        and citation.source_uri
        and citation.strategy_id
        and citation.end_char > citation.start_char >= 0
    )


def run(args: argparse.Namespace) -> dict:
    package = assert_installed_package(
        expected_version=args.expected_package_version,
        forbidden_source_root=args.forbid_source_root,
    )
    public = chunk("public", "mars public operations evidence")
    private = chunk(
        "private",
        "mars private restricted evidence",
        policy=PermissionPolicy(
            tenant_id="tenant-a",
            allowed_groups=frozenset({"ops"}),
        ),
    )
    excluded = chunk("excluded", "mars excluded evidence")
    scores = {
        public.source_uri: 0.1,
        private.source_uri: 100.0,
        excluded.source_uri: 50.0,
    }

    public_reranker = RecordingReranker(scores)
    public_retriever = build_retriever((public, private, excluded), public_reranker)
    public_query = RetrievalQuery(
        text="mars evidence",
        top_k=10,
        excluded_chunk_ids=frozenset({excluded.chunk_id}),
    )
    public_results_a = public_retriever.search(public_query)
    public_results_b = public_retriever.search(public_query)
    public_ids_a = [result.chunk.chunk_id for result in public_results_a]
    public_ids_b = [result.chunk.chunk_id for result in public_results_b]

    authorization_leakage_count = sum(
        result.chunk.chunk_id == private.chunk_id for result in public_results_a
    )
    excluded_chunk_leakage_count = sum(
        result.chunk.chunk_id == excluded.chunk_id for result in public_results_a
    )
    unauthorized_scoring_count = sum(
        uri == private.source_uri for uri in public_reranker.scored_source_uris
    )

    authorized_reranker = RecordingReranker(scores)
    authorized_retriever = build_retriever(
        (public, private, excluded), authorized_reranker
    )
    authorized_results = authorized_retriever.search(
        RetrievalQuery(
            text="mars restricted evidence",
            top_k=10,
            principal=AccessPrincipal(
                tenant_id="tenant-a",
                groups=frozenset({"ops"}),
            ),
            excluded_chunk_ids=frozenset({excluded.chunk_id}),
        )
    )
    authorized_private_visible = any(
        result.chunk.chunk_id == private.chunk_id for result in authorized_results
    )

    all_results = [*public_results_a, *authorized_results]
    citation_completeness = (
        sum(citation_complete(result) for result in all_results) / len(all_results)
        if all_results
        else 0.0
    )

    malformed = chunk(
        "malformed",
        "mars malformed protected evidence",
        metadata={GROUPS_METADATA_KEY: "ops,,admin"},
    )
    malformed_policy_fails_closed = False
    try:
        build_retriever((malformed,), RecordingReranker({malformed.source_uri: 1.0}))
    except ValueError:
        malformed_policy_fails_closed = True

    output = {
        "schema_version": "atlasrag.promotion-contract-evaluation.v1",
        "protocol_id": args.protocol_id,
        "package": {
            **package,
            "git_commit": args.atlasrag_git_commit,
            "wheel": args.wheel.name,
            "wheel_sha256": sha256(args.wheel),
        },
        "contracts": {
            "authorization_leakage_count": authorization_leakage_count,
            "excluded_chunk_leakage_count": excluded_chunk_leakage_count,
            "unauthorized_candidate_scoring_count": unauthorized_scoring_count,
            "malformed_policy_fails_closed": malformed_policy_fails_closed,
            "citation_completeness": citation_completeness,
            "deterministic_ranking_reproduction": public_ids_a == public_ids_b,
            "authorized_private_visible": authorized_private_visible,
        },
        "scope": "fully synthetic installed-package authorization, exclusion, citation, and determinism contract evaluation",
        "payload_redistributed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    required = (
        authorization_leakage_count == 0
        and excluded_chunk_leakage_count == 0
        and unauthorized_scoring_count == 0
        and malformed_policy_fails_closed
        and citation_completeness == 1.0
        and public_ids_a == public_ids_b
        and authorized_private_visible
    )
    if not required:
        raise SystemExit(2)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--atlasrag-git-commit", required=True)
    parser.add_argument("--expected-package-version", default="0.3.0.dev0")
    parser.add_argument("--forbid-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
