# Minimal performance harness

This directory contains a small, process-isolated performance harness for **local repository algorithms only**.

It does **not** integrate or call an AI service, external model API, Agent, API key, or network inference. The optional CUDA case runs the repository's already-existing local `torch_mlp` implementation; if Torch/CUDA is unavailable it is reported as `SKIP`.

## Features

- one fresh Python process per case, before heavy imports;
- explicit warm-up and measured repeats;
- wall/CPU-time samples with linear-interpolated P50/P95;
- current-process peak RSS via stdlib (`psutil` is used when available);
- optional CUDA allocated/reserved peak memory;
- host, package, thread and capability `manifest.json`;
- deterministic SHA-256 consistency digest per measured result;
- raw per-repeat samples and a compact `summary.csv/json`;
- best-effort process-tree cleanup when a case times out;
- default output only under `benchmarks/perf/results/`, never `runs/` or `benchmarks/results/`.

## Quick start

From the repository root:

```powershell
# list cases; no heavy dependency import
python benchmarks/perf/harness.py --list

# default quick profile: NumPy linear algebra + random-search optimization
python benchmarks/perf/harness.py --cases quick --warmup 1 --repeats 5

# all locally available cases; optional UMAP/Torch cases are skipped if absent
python benchmarks/perf/harness.py --cases all --warmup 1 --repeats 7

# include local torch CPU, and CUDA only when capability probe finds it
python benchmarks/perf/harness.py --cases cpu_linalg,cpu_rf,torch_cpu `
  --include-gpu --threads 1 --warmup 2 --repeats 10
```

For a bounded parallel test of the toolbox RandomForest:

```powershell
$env:MLTB_NJOBS="1"
python benchmarks/perf/harness.py --cases cpu_rf --n-jobs 1 `
  --threads 1 --warmup 1 --repeats 5

$env:MLTB_NJOBS="2"
python benchmarks/perf/harness.py --cases cpu_rf --n-jobs 2 `
  --threads 1 --warmup 1 --repeats 5
```

Use a fresh process for every thread configuration. The harness already does this when cases/options are run as separate commands.

## Cases

| Case | Meaning | Default | Requirements |
|---|---|---:|---|
| `cpu_linalg` | NumPy matrix multiply; BLAS/OpenMP smoke | yes | NumPy |
| `opt_random` | optimization registry + random-search ask/tell loop | yes | project core/opt |
| `cpu_rf` | toolbox RandomForest via `core.runner` | no | scikit-learn/project methods |
| `numba_umap` | local UMAP and Numba warm-up path | no | `umap-learn`, Numba |
| `torch_cpu` | existing local `torch_mlp`, explicit CPU | no | optional Torch |
| `torch_cuda` | existing local `torch_mlp`, explicit CUDA | no | optional CUDA-capable Torch |

`--cases` accepts `quick`, `all`, or a comma-separated list. `--include-optional` adds optional CPU cases; `--include-gpu` adds `torch_cuda` when available.

## Output layout

```text
benchmarks/perf/results/<UTC>-pid<pid>/
├── manifest.json
├── summary.json
├── summary.csv
├── 01-cpu_linalg.json
└── 02-opt_random.json
```

A case JSON contains:

- `prepare_s`;
- warm-up records (excluded from measured P50/P95);
- every measured wall/CPU/RSS sample;
- wall/CPU/RSS/GPU statistics including P50/P95;
- a deterministic result digest and consistency verdict.

`manifest.json` records Python/OS/CPU/RAM, package versions, capability probes, thread variables, warm-up/repeat counts, and explicit skips. It also records best-effort read-only git commit/dirty state, SHA-256 for all local `ml_toolbox` Python sources plus the harness/cases/requirements, and per-case immutable input hashes. Git failure is recorded as `available: false` rather than failing the benchmark. It intentionally makes no network calls.

## Consistency summary

Each case returns a deterministic payload such as predictions, metrics, parameters, embedding, or a matrix checksum. The harness hashes dtype/shape/bytes recursively and reports:

- `stable=true` when all measured digests match;
- the number of unique measured digests;
- whether the last warm-up matches the first measured result.

Randomized algorithms use fixed seeds. GPU and different Numba-thread profiles may legitimately require looser scientific tolerances in a future acceptance layer; this minimal harness first exposes the exact result differences.

## Memory and timing notes

- timing uses `time.perf_counter_ns()`;
- process CPU time uses `time.process_time_ns()`;
- the CUDA case synchronizes before and after measured work;
- RSS sampling is current-process only in this minimal version; a process-tree sampler can be added without changing the case interface;
- P95 is an observed sample quantile, so use at least 20 repeats for a strict gate.

## Self-test

The fast self-test checks percentile math, digest stability, payload summaries, case capability selection, and a two-repeat NumPy worker run:

```powershell
python benchmarks/perf/test_harness.py
```

A one-case end-to-end run is:

```powershell
python benchmarks/perf/harness.py --cases cpu_linalg --warmup 0 --repeats 2
```
