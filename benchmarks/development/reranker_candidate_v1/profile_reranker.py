"""Synthetic-only reranker component profiler for the frozen development protocol."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

try:
    from benchmarks.development.reranker_candidate_v1.verify_protocol import validate
except ModuleNotFoundError:
    from verify_protocol import validate

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "atlasrag-reranker-candidate-v1.lock"


class Predictor(Protocol):
    """Minimal CrossEncoder-compatible prediction surface."""

    def predict(self, pairs: Sequence[tuple[str, str]], *, batch_size: int) -> Any: ...


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError("invalid frozen protocol: " + "; ".join(errors))
    return protocol


def _hub_root() -> Path:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def expected_snapshot_path(protocol: dict[str, object]) -> Path:
    reranker = protocol["reranker"]
    assert isinstance(reranker, dict)
    name = str(reranker["name"])
    revision = str(reranker["revision"])
    model_dir = "models--" + name.replace("/", "--")
    return _hub_root() / model_dir / "snapshots" / revision


def require_local_snapshot(protocol: dict[str, object]) -> Path:
    snapshot = expected_snapshot_path(protocol)
    if not snapshot.is_dir():
        raise RuntimeError(
            f"required pinned local model snapshot is missing: {snapshot}"
        )
    return snapshot


@contextlib.contextmanager
def offline_model_environment() -> Iterator[None]:
    """Force Hugging Face model resolution into local/offline mode."""

    keys = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def exclusive_benchmark_lock(path: Path = DEFAULT_LOCK_PATH) -> Iterator[Path]:
    """Acquire the required single-profiler OS lock without waiting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"exclusive profiler lock is already held: {path}"
            ) from exc
        acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield path
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        if acquired:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


def generate_synthetic_inputs(
    protocol: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Generate the declared deterministic workload without reading benchmark data."""

    source = protocol["input_source"]
    matrix = protocol["matrix"]
    assert isinstance(source, dict)
    assert isinstance(matrix, dict)
    seed = int(source["seed"])
    query_count = int(source["query_count"])
    max_depth = max(int(value) for value in matrix["candidate_depths"])
    rng = random.Random(seed)
    vocabulary = (
        "amber",
        "beacon",
        "cobalt",
        "delta",
        "ember",
        "fjord",
        "glyph",
        "harbor",
        "ion",
        "jigsaw",
        "kepler",
        "lattice",
        "mesa",
        "nova",
        "orbit",
        "prism",
        "quartz",
        "relay",
        "saffron",
        "tundra",
        "ultra",
        "vector",
        "willow",
        "zenith",
    )
    rows: list[dict[str, object]] = []
    for query_index in range(query_count):
        query_terms = rng.sample(vocabulary, 4)
        query = f"synthetic retrieval component query {query_index:02d} " + " ".join(
            query_terms
        )
        documents: list[dict[str, str]] = []
        for document_index in range(max_depth):
            overlap = 4 - (document_index % 5)
            kept = query_terms[: max(0, overlap)]
            filler = rng.sample(vocabulary, 6)
            text = (
                f"synthetic component document {query_index:02d}-{document_index:02d}; "
                f"topic {' '.join(kept)}; control {' '.join(filler)}; "
                "generated solely for reranker component profiling"
            )
            documents.append(
                {
                    "id": f"synthetic-{query_index:02d}-{document_index:02d}",
                    "text": text,
                }
            )
        rows.append(
            {"id": f"query-{query_index:02d}", "query": query, "documents": documents}
        )

    forbidden = {
        str(value).casefold() for value in protocol["prohibited_dataset_names"]
    }
    serialized = json.dumps(rows, sort_keys=True).casefold()
    present = sorted(name for name in forbidden if name in serialized)
    if present:
        raise RuntimeError(
            f"synthetic workload contains prohibited dataset names: {present}"
        )
    return tuple(rows)


def workload_sha256(rows: Sequence[dict[str, object]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pairs(
    rows: Sequence[dict[str, object]], depth: int
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        query = str(row["query"])
        documents = row["documents"]
        assert isinstance(documents, list)
        if len(documents) < depth:
            raise RuntimeError(
                "synthetic row is shorter than requested candidate depth"
            )
        pairs.extend((query, str(document["text"])) for document in documents[:depth])
    return tuple(pairs)


def _coerce_scores(raw: Any) -> tuple[float, ...]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return tuple(float(value) for value in raw)


def validate_scores(scores: Sequence[float], expected_count: int) -> None:
    if len(scores) != expected_count:
        raise RuntimeError(
            f"score count mismatch: expected {expected_count}, observed {len(scores)}"
        )
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("reranker produced non-finite scores")


def ordering_digest(
    rows: Sequence[dict[str, object]], scores: Sequence[float], depth: int
) -> str:
    expected = len(rows) * depth
    validate_scores(scores, expected)
    orders: list[list[str]] = []
    offset = 0
    for row in rows:
        documents = row["documents"]
        assert isinstance(documents, list)
        segment = scores[offset : offset + depth]
        indexed = list(zip(documents[:depth], segment, strict=True))
        indexed.sort(key=lambda item: (-item[1], str(item[0]["id"])))
        orders.append([str(document["id"]) for document, _ in indexed])
        offset += depth
    payload = json.dumps(orders, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_metadata(device: str) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "device": device,
        "dependencies": {
            "sentence-transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
    }


def resolve_device() -> tuple[str, Any | None]:
    try:
        import torch
    except ImportError:
        return "cpu", None
    if torch.cuda.is_available():
        return "cuda:0", torch
    return "cpu", torch


def load_cross_encoder(protocol: dict[str, object], device: str) -> Predictor:
    reranker = protocol["reranker"]
    assert isinstance(reranker, dict)
    require_local_snapshot(protocol)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "real profiling requires the AtlasRAG 'reranking' extra; "
            "sentence-transformers is not installed"
        ) from exc
    with offline_model_environment():
        return CrossEncoder(
            str(reranker["name"]),
            revision=str(reranker["revision"]),
            device=device,
            local_files_only=True,
        )


def _synchronizer(device: str, torch_module: Any | None) -> Callable[[], None]:
    if device.startswith("cuda"):
        if torch_module is None or not torch_module.cuda.is_available():
            raise RuntimeError("CUDA device selected without an available CUDA runtime")
        return torch_module.cuda.synchronize
    return lambda: None


def _reset_peak_memory(device: str, torch_module: Any | None) -> None:
    if device.startswith("cuda"):
        assert torch_module is not None
        torch_module.cuda.reset_peak_memory_stats()


def _peak_memory_mib(device: str, torch_module: Any | None) -> float | None:
    if not device.startswith("cuda"):
        return None
    assert torch_module is not None
    return float(torch_module.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def profile_configuration(
    predictor: Predictor,
    rows: Sequence[dict[str, object]],
    *,
    depth: int,
    batch_size: int,
    warmups: int,
    measured_iterations: int,
    synchronize: Callable[[], None],
    peak_memory_reset: Callable[[], None] = lambda: None,
    peak_memory_read: Callable[[], float | None] = lambda: None,
) -> dict[str, object]:
    pairs = build_pairs(rows, depth)
    expected_count = len(pairs)
    for _ in range(warmups):
        scores = _coerce_scores(predictor.predict(pairs, batch_size=batch_size))
        validate_scores(scores, expected_count)

    first = _coerce_scores(predictor.predict(pairs, batch_size=batch_size))
    second = _coerce_scores(predictor.predict(pairs, batch_size=batch_size))
    first_digest = ordering_digest(rows, first, depth)
    second_digest = ordering_digest(rows, second, depth)
    deterministic = first_digest == second_digest
    if not deterministic:
        raise RuntimeError(
            "repeat ordering determinism failed for "
            f"depth={depth}, batch_size={batch_size}"
        )

    peak_memory_reset()
    samples: list[float] = []
    for _ in range(measured_iterations):
        synchronize()
        started = time.perf_counter()
        scores = _coerce_scores(predictor.predict(pairs, batch_size=batch_size))
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        validate_scores(scores, expected_count)
        samples.append(elapsed_ms)

    return {
        "candidate_depth": depth,
        "batch_size": batch_size,
        "pair_count_per_iteration": expected_count,
        "warmup_iterations": warmups,
        "measured_iterations": measured_iterations,
        "raw_timing_ms": samples,
        "timing_ms": {
            "p50": statistics.median(samples),
            "p95": percentile(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
        },
        "repeat_order_deterministic": deterministic,
        "ordering_sha256": first_digest,
        "peak_accelerator_memory_mib": peak_memory_read(),
    }


def nominate_candidate(
    results: Sequence[dict[str, object]], protocol: dict[str, object]
) -> dict[str, object] | None:
    gates = protocol["candidate_gates"]
    assert isinstance(gates, dict)
    depth = int(gates["default_candidate_depth"])
    allowed_ratio = float(gates["batch_size_p95_over_best_same_depth_max_ratio"])
    memory_limit = float(gates["peak_accelerator_memory_mib_max"])
    eligible: list[dict[str, object]] = []
    for result in results:
        if int(result["candidate_depth"]) != depth:
            continue
        if result["repeat_order_deterministic"] is not True:
            continue
        peak = result["peak_accelerator_memory_mib"]
        if peak is not None and float(peak) > memory_limit:
            continue
        eligible.append(result)
    if not eligible:
        return None
    best_p95 = min(float(result["timing_ms"]["p95"]) for result in eligible)  # type: ignore[index]
    tolerance = best_p95 * allowed_ratio
    within_tolerance = [
        result
        for result in eligible
        if float(result["timing_ms"]["p95"]) <= tolerance  # type: ignore[index]
    ]
    chosen = min(
        within_tolerance,
        key=lambda result: (
            int(result["batch_size"]),
            float(result["timing_ms"]["p95"]),  # type: ignore[index]
        ),
    )
    return {
        "candidate_depth": int(chosen["candidate_depth"]),
        "batch_size": int(chosen["batch_size"]),
        "selection_basis": (
            "frozen default depth; smallest batch size within 1.10x of the best "
            "same-depth p95 after determinism and memory gates"
        ),
        "best_same_depth_p95_ms": best_p95,
        "chosen_p95_ms": float(chosen["timing_ms"]["p95"]),  # type: ignore[index]
    }


def run_profile(
    protocol: dict[str, object],
    predictor: Predictor,
    *,
    device: str,
    torch_module: Any | None,
) -> dict[str, object]:
    rows = generate_synthetic_inputs(protocol)
    matrix = protocol["matrix"]
    assert isinstance(matrix, dict)
    synchronize = _synchronizer(device, torch_module)
    results: list[dict[str, object]] = []
    for depth in matrix["candidate_depths"]:
        for batch_size in matrix["batch_sizes"]:
            result = profile_configuration(
                predictor,
                rows,
                depth=int(depth),
                batch_size=int(batch_size),
                warmups=int(matrix["warmup_iterations"]),
                measured_iterations=int(matrix["measured_iterations"]),
                synchronize=synchronize,
                peak_memory_reset=lambda: _reset_peak_memory(device, torch_module),
                peak_memory_read=lambda: _peak_memory_mib(device, torch_module),
            )
            results.append(result)
    candidate = nominate_candidate(results, protocol)
    if candidate is None:
        raise RuntimeError("no candidate configuration passed the development gates")
    return {
        "schema_version": "atlasrag.reranker-development-profile.v1",
        "status": "development_profile_complete",
        "frozen_protocol_declared_status": protocol["status"],
        "input_kind": "deterministic_synthetic_templates",
        "query_count": len(rows),
        "workload_sha256": workload_sha256(rows),
        "environment": environment_metadata(device),
        "configurations": results,
        "nominated_candidates": [candidate],
        "claim_boundary": protocol["claim_boundary"],
        "limitations": [
            (
                "Synthetic component profiling does not establish natural-workload "
                "quality."
            ),
            "Timing is a controlled single-host observation, not a production SLO.",
            (
                "The frozen final SciFact and ArguAna payloads are not inputs to "
                "this profiler."
            ),
            (
                "A separately frozen final evaluation is required before any "
                "promotion decision."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    require_local_snapshot(protocol)
    device, torch_module = resolve_device()
    with exclusive_benchmark_lock(args.lock_path):
        predictor = load_cross_encoder(protocol, device)
        report = run_profile(
            protocol,
            predictor,
            device=device,
            torch_module=torch_module,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: wrote synthetic-only development profile to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
