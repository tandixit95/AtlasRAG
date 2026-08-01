# AtlasRAG Benchmark Reproduction Runbook

All commands are local. They do not publish, deploy, create a DOI, or redistribute dataset payloads.

## 1. Environment

```bash
python3.12 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install 'pytest>=8.3,<9' 'ruff>=0.12,<1' build
```

## 2. Acquire data from the recorded sources

Download SciFact and ArguAna from the source URLs in `DATASET_PROVENANCE.json`, then verify the archive SHA-256 values before extraction. Do not publish the archives or extracted payloads as part of the release candidate.

Create the deterministic ArguAna contrast slice with the frozen selection rule and verify `artifacts/datasets/arguana-contrast-200/SLICE_MANIFEST.json`.

## 3. Neutral harness

```bash
python src/run_batched_benchmark.py \
  --dataset scifact \
  --data-dir artifacts/datasets/scifact \
  --split test \
  --workdir artifacts/cache/official-scifact \
  --output artifacts/scifact-official-a.json \
  --candidate-k 100 --rrf-k 60 --ann-ef 100 \
  --latency-sample 25 --seed 20260731 --rebuild

python src/run_batched_benchmark.py \
  --dataset arguana \
  --data-dir artifacts/datasets/arguana-contrast-200 \
  --split test \
  --workdir artifacts/cache/official-arguana \
  --output artifacts/arguana-contrast-official-a.json \
  --candidate-k 100 --rrf-k 60 --ann-ef 100 \
  --latency-sample 25 --seed 20260731 --rebuild
```

Repeat without `--rebuild` into the corresponding `-b.json` outputs, then run the neutral reproducibility checker.

## 4. Build and install AtlasRAG

From the approved AtlasRAG source commit:

```bash
python -m build
ATLASRAG_WHEEL=$(find dist -maxdepth 1 -name 'atlasrag-0.2.0-*.whl' -print -quit)
python -m pip install --force-reinstall "$ATLASRAG_WHEEL"
sha256sum "$ATLASRAG_WHEEL"
```

The approved wheel SHA-256 is:

`30cbaf0030fe86177b7962e43267b6d182534c023eb9d61e7eec7481df048200`

## 5. Installed-package benchmark

Set the immutable local model snapshot and package identity:

```bash
export MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
export MODEL_SNAPSHOT="$HF_HOME/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/$MODEL_REVISION"
export ATLASRAG_COMMIT=5e86c78a4c40bc6d552d14d4fdcc370b0db8ece1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
```

Run A with cache rebuild and B from the frozen cache for each dataset:

```bash
python src/run_installed_atlasrag_benchmark.py \
  --dataset scifact \
  --data-dir artifacts/datasets/scifact \
  --output artifacts/scifact-atlasrag-installed-a.json \
  --embedding-cache artifacts/cache/installed-atlasrag-scifact/corpus_embeddings.npy \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-revision "$MODEL_REVISION" \
  --wheel "$ATLASRAG_WHEEL" \
  --atlasrag-git-commit "$ATLASRAG_COMMIT" \
  --forbid-source-root "$ATLASRAG_SOURCE" \
  --rebuild-embeddings

python src/run_installed_atlasrag_benchmark.py \
  --dataset arguana \
  --data-dir artifacts/datasets/arguana-contrast-200 \
  --output artifacts/arguana-contrast-atlasrag-installed-a.json \
  --embedding-cache artifacts/cache/installed-atlasrag-arguana/corpus_embeddings.npy \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-revision "$MODEL_REVISION" \
  --wheel "$ATLASRAG_WHEEL" \
  --atlasrag-git-commit "$ATLASRAG_COMMIT" \
  --forbid-source-root "$ATLASRAG_SOURCE" \
  --rebuild-embeddings
```

Repeat into the `-b.json` outputs without `--rebuild-embeddings`.

## 6. Package reproducibility comparison

```bash
python src/check_installed_package_results.py \
  --scifact-a artifacts/scifact-atlasrag-installed-a.json \
  --scifact-b artifacts/scifact-atlasrag-installed-b.json \
  --scifact-neutral artifacts/scifact-official-a.json \
  --arguana-a artifacts/arguana-contrast-atlasrag-installed-a.json \
  --arguana-b artifacts/arguana-contrast-atlasrag-installed-b.json \
  --arguana-neutral artifacts/arguana-contrast-official-a.json \
  --output artifacts/installed-package-reproducibility.json
```

## 7. Safety and validation

```bash
python src/run_safety_eval.py \
  --dataset artifacts/datasets/synthetic_reliability_v1.json \
  --output artifacts/safety-evaluation.json

python src/run_atlasrag_adapter_smoke.py \
  --dataset artifacts/datasets/synthetic_reliability_v1.json \
  --output artifacts/atlasrag-adapter-smoke.json

python -m unittest discover -s tests -v
python -m ruff check src tests
python src/validate_package.py
python src/check_regression_gates.py \
  --config REGRESSION_GATES.json \
  --output artifacts/regression-gate-report.json
```

## 8. Checksum and privacy preflight

Regenerate `SHA256SUMS` only after all accepted artifacts are final. Exclude local caches, virtual environments, dataset archives, extracted third-party records, and private-path manifests from the public package. Scan public text and release artifacts for credentials, private paths, employer terms, smart punctuation, and unresolved placeholders.
