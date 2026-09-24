# -*- coding: utf-8 -*-
"""Fast self-tests for benchmarks/perf/harness.py.

Run from the repository root:
    python benchmarks/perf/test_harness.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness
from cases import TorchMlpCase


def test_percentile() -> None:
    assert harness.percentile([], 0.95) is None
    assert harness.percentile([2.0], 0.95) == 2.0
    assert harness.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert abs(harness.percentile(list(range(1, 21)), 0.95) - 19.05) < 1e-12
    stats = harness._stats([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4 and stats["p50"] == 2.5


def test_stable_digest() -> None:
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = a.copy()
    assert harness.stable_digest({"a": a}) == harness.stable_digest({"a": b})
    b[0, 0] += 1
    assert harness.stable_digest(a) != harness.stable_digest(b)
    assert harness.stable_digest({"x": [1, 2]}) == harness.stable_digest({"x": (1, 2)})


def test_payload_summary() -> None:
    payload = {
        "metrics": {"f1": 0.9}, "finite": True,
        "gpu_allocated_peak_bytes": 1024 ** 2,
        "gpu_reserved_peak_bytes": 2 * 1024 ** 2,
    }
    out = harness._payload_summary(payload)
    assert len(out["digest"]) == 64
    assert out["metrics"]["f1"] == 0.9
    assert out["gpu_allocated_peak_bytes_mb"] == 1.0
    assert out["gpu_reserved_peak_bytes_mb"] == 2.0
    other = dict(payload, gpu_allocated_peak_bytes=2 * 1024 ** 2)
    assert out["digest"] == harness._payload_summary(other)["digest"]


def test_case_resolution() -> None:
    args = argparse.Namespace(include_optional=False, include_gpu=False)
    selected, skipped = harness._resolve_cases("quick", {"umap": False}, args)
    assert selected == ["cpu_linalg", "opt_random"] and not skipped
    selected, skipped = harness._resolve_cases("all", {"umap": False, "torch": False}, args)
    assert "numba_umap" not in selected and "torch_cpu" not in selected
    assert {x["case_id"] for x in skipped} >= {"numba_umap", "torch_cpu", "torch_cuda"}


def test_source_provenance_helpers() -> None:
    hashes = harness.source_hashes()
    assert "benchmarks/perf/harness.py" in hashes
    assert "benchmarks/perf/cases.py" in hashes
    assert "ml_toolbox/core/parallel.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_torch_cpu_prepare_does_not_require_cuda() -> None:
    fake_torch = types.SimpleNamespace()
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        case = TorchMlpCase({"torch_samples": 64, "torch_features": 8})
        case.prepare()
    assert case.device == "cpu"
    assert len(case.input_sha256()) == 64


def test_worker_quick_case() -> None:
    with tempfile.TemporaryDirectory(prefix="mltb_perf_") as td:
        path = Path(td) / "cpu_linalg.json"
        rec = harness.run_worker("cpu_linalg", path,
                                 {"linalg_size": 32, "linalg_repeats": 1},
                                 warmup=0, repeats=2)
        assert rec["status"] == "ok", rec
        assert rec["stats"]["wall_s"]["n"] == 2
        assert rec["consistency"]["stable"] is True
        assert len(rec["input_sha256"]) == 64
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["status"] == "ok"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
