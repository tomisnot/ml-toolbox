# -*- coding: utf-8 -*-
"""Minimal, process-isolated algorithm performance harness.

Default output is kept under ``benchmarks/perf/results``; it never writes ML/opt
business run records.  The parent launches one fresh Python process per case so
OpenMP, joblib, Numba, BLAS and Torch thread/device settings are established before
heavy imports.

Examples
--------
    python benchmarks/perf/harness.py --list
    python benchmarks/perf/harness.py --cases quick --warmup 1 --repeats 5
    python benchmarks/perf/harness.py --cases all --include-gpu --warmup 1 --repeats 7
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The harness must not create __pycache__ artifacts in the measured workspace.
sys.dont_write_bytecode = True

from cases import CASE_BY_ID, CASE_SPECS, make_case  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PERF_ROOT = HERE
DEFAULT_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "numba",
    "umap-learn", "torch", "xgboost", "lightgbm", "psutil",
)
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
)


# --------------------------------------------------------------------------- utils
def _jsonable(value: Any) -> Any:
    """Convert common numeric/container values to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; q is in [0, 1]."""
    if not values:
        return None
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "min": None, "p50": None, "p95": None,
                "max": None, "mean": None, "std": None, "cv": None}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    return {
        "n": len(xs), "min": min(xs), "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95), "max": max(xs), "mean": mean,
        "std": std, "cv": (std / mean if mean else None),
    }


def _normalize_for_digest(value: Any):
    """Return deterministic bytes for numpy/scalar/container result payloads."""
    if value is None:
        return b"none"
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        if math.isnan(value):
            token = "nan"
        elif math.isinf(value):
            token = "inf+" if value > 0 else "inf-"
        else:
            token = value.hex()
        return b"float:" + token.encode("ascii")
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, dict):
        h = hashlib.sha256()
        for key in sorted(value, key=str):
            h.update(str(key).encode("utf-8"))
            h.update(b"=")
            h.update(stable_digest(value[key]).encode("ascii"))
            h.update(b";")
        return b"dict:" + h.digest()
    if isinstance(value, (list, tuple)):
        h = hashlib.sha256()
        for item in value:
            h.update(stable_digest(item).encode("ascii"))
            h.update(b";")
        return b"seq:" + h.digest()
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        arr = value
        try:
            if not arr.flags.c_contiguous:
                arr = arr.copy(order="C")
            raw = arr.tobytes(order="C")
        except TypeError:
            raw = arr.tobytes()
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(raw)
        return b"array:" + h.digest()
    if hasattr(value, "item"):
        return stable_digest(value.item())
    return stable_digest(str(value))


def stable_digest(value: Any) -> str:
    return hashlib.sha256(_normalize_for_digest(value)).hexdigest()


# --------------------------------------------------------------------------- memory
def _rss_mb() -> float | None:
    """Current process RSS. Uses stdlib; psutil is optional when installed."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 2)
    except Exception:
        pass
    if os.name == "nt":
        try:
            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters), counters.cb,
            )
            return float(counters.WorkingSetSize) / (1024 ** 2) if ok else None
        except Exception:
            return None
    try:
        import resource  # type: ignore
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux/BSD report KiB; macOS reports bytes.
        return raw / (1024 ** 2) if os.name == "darwin" else raw / 1024.0
    except Exception:
        return None


class _PeakRss:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = _rss_mb()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            current = _rss_mb()
            if current is not None:
                self.peak = max(self.peak or 0.0, current)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval * 4))
        current = _rss_mb()
        if current is not None:
            self.peak = max(self.peak or 0.0, current)


# --------------------------------------------------------------------------- manifest
def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in DEFAULT_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _total_memory_mb() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total // (1024 ** 2))
    except Exception:
        pass
    if os.name == "nt":
        try:
            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys // (1024 ** 2))
        except Exception:
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 ** 2))
    except Exception:
        return None


def capabilities() -> dict[str, Any]:
    """Installed/usable optional capabilities; no network or model service calls."""
    out: dict[str, Any] = {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "umap": importlib.util.find_spec("umap") is not None,
        "numba": importlib.util.find_spec("numba") is not None,
        "torch": False,
        "torch_cuda": False,
        "gpu_name": None,
        "torch_cuda_version": None,
        "numba_threads": None,
    }
    if out["numba"]:
        try:
            import numba  # type: ignore
            out["numba_threads"] = int(numba.get_num_threads())
        except Exception:
            pass
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore
            out["torch"] = True
            out["torch_cuda_version"] = torch.version.cuda
            out["torch_cuda"] = bool(torch.cuda.is_available())
            if out["torch_cuda"]:
                out["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            out["torch_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _git_state() -> dict[str, Any]:
    """Read-only git identity/state; failure is data, not a harness failure."""
    state: dict[str, Any] = {
        "available": False, "commit": None, "dirty": None,
        "status_entries": None, "error": None,
    }
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=5, check=False,
        )
        if rev.returncode != 0:
            state["error"] = (rev.stderr or rev.stdout or "git rev-parse failed").strip()[:300]
            return state
        state["commit"] = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=10, check=False,
        )
        if status.returncode == 0:
            lines = [line for line in status.stdout.splitlines() if line.strip()]
            state.update({"available": True, "dirty": bool(lines),
                          "status_entries": lines[:100]})
        else:
            state["commit_available"] = True
            state["error"] = (status.stderr or status.stdout or "git status failed").strip()[:300]
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def source_hashes() -> dict[str, str]:
    """Hash the harness plus all local Python sources cases can execute."""
    paths = [HERE / "harness.py", HERE / "cases.py",
             PROJECT_ROOT / "requirements.txt"]
    paths.extend(sorted((PROJECT_ROOT / "ml_toolbox").rglob("*.py")))
    out = {}
    for path in paths:
        if path.exists() and path.is_file():
            out[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = \
                hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local algorithms only; no AI/API/Agent/network integration",
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpus": os.cpu_count(),
            "total_memory_mb": _total_memory_mb(),
        },
        "packages": _package_versions(),
        "capabilities": capabilities(),
        "git": _git_state(),
        "environment": {k: os.environ.get(k) for k in THREAD_ENV + ("MLTB_NJOBS",)},
        "settings": {
            "cases": args.cases,
            "include_optional": bool(args.include_optional),
            "include_gpu": bool(args.include_gpu),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "threads": int(args.threads),
            "n_jobs": args.n_jobs,
            "numba_threads": int(args.numba_threads),
            "case_timeout_s": float(args.case_timeout),
        },
        "source_sha256": source_hashes(),
    }


# --------------------------------------------------------------------------- worker
def _payload_summary(payload: Any) -> dict[str, Any]:
    # Resource counters may fluctuate even when scientific output is identical;
    # keep them in the summary but exclude them from the consistency digest.
    digest_payload = payload
    if isinstance(payload, dict):
        digest_payload = {k: v for k, v in payload.items()
                          if not str(k).startswith("gpu_")}
    summary: dict[str, Any] = {"digest": stable_digest(digest_payload)}
    if isinstance(payload, dict):
        for key in ("metrics", "params", "device", "finite", "shape", "trustworthiness"):
            if key in payload:
                summary[key] = _jsonable(payload[key])
        for key in ("gpu_allocated_peak_bytes", "gpu_reserved_peak_bytes"):
            if key in payload:
                summary[f"{key}_mb"] = float(payload[key]) / (1024 ** 2)
    return summary


def run_worker(case_id: str, output_path: Path, config: dict[str, Any],
               warmup: int, repeats: int) -> dict[str, Any]:
    started = time.time()
    case = None
    try:
        case = make_case(case_id, config)
        prepare_t0 = time.perf_counter_ns()
        case.prepare()
        prepare_s = (time.perf_counter_ns() - prepare_t0) / 1e9
        input_sha256 = case.input_sha256()

        warmups = []
        for i in range(warmup):
            t0 = time.perf_counter_ns()
            payload = case.run()
            warmups.append({
                "index": i,
                "wall_s": (time.perf_counter_ns() - t0) / 1e9,
                "result": _payload_summary(payload),
            })

        samples = []
        baseline_rss_mb = _rss_mb()
        for i in range(repeats):
            cpu0 = time.process_time_ns()
            wall0 = time.perf_counter_ns()
            with _PeakRss() as mem:
                payload = case.run()
            wall_s = (time.perf_counter_ns() - wall0) / 1e9
            cpu_s = (time.process_time_ns() - cpu0) / 1e9
            samples.append({
                "index": i,
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "rss_peak_mb": mem.peak,
                "result": _payload_summary(payload),
            })

        case.close()
        measured_digests = [s["result"]["digest"] for s in samples]
        warmup_digests = [w["result"]["digest"] for w in warmups]
        result = {
            "schema": 1,
            "status": "ok",
            "case_id": case_id,
            "input_sha256": input_sha256,
            "started_unix": started,
            "elapsed_s": time.time() - started,
            "prepare_s": prepare_s,
            "baseline_rss_mb": baseline_rss_mb,
            "warmup": warmups,
            "warmup_stats": {"wall_s": _stats([w["wall_s"] for w in warmups])},
            "samples": samples,
            "stats": {
                "wall_s": _stats([s["wall_s"] for s in samples]),
                "cpu_s": _stats([s["cpu_s"] for s in samples]),
                "rss_peak_mb": _stats([s["rss_peak_mb"] for s in samples
                                       if s["rss_peak_mb"] is not None]),
                "gpu_allocated_peak_mb": _stats([
                    s["result"].get("gpu_allocated_peak_bytes_mb")
                    for s in samples
                    if s["result"].get("gpu_allocated_peak_bytes_mb") is not None]),
                "gpu_reserved_peak_mb": _stats([
                    s["result"].get("gpu_reserved_peak_bytes_mb")
                    for s in samples
                    if s["result"].get("gpu_reserved_peak_bytes_mb") is not None]),
            },
            "consistency": {
                "stable": bool(measured_digests) and len(set(measured_digests)) == 1,
                "measured_digest_count": len(set(measured_digests)),
                "warmup_matches_measured": (
                    not warmup_digests or (measured_digests and
                    warmup_digests[-1] == measured_digests[0])),
                "unique_measured_digests": sorted(set(measured_digests)),
            },
        }
    except Exception as exc:
        try:
            if case is not None:
                case.close()
        except Exception:
            pass
        result = {
            "schema": 1, "status": "error", "case_id": case_id,
            "started_unix": started, "elapsed_s": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    _atomic_json(output_path, result)
    return result


# --------------------------------------------------------------------------- parent
def _resolve_cases(requested: str, caps: dict[str, Any], args) -> tuple[list[str], list[dict]]:
    known = list(CASE_BY_ID)
    if requested == "quick":
        selected = [s.case_id for s in CASE_SPECS if s.default]
    elif requested == "all":
        selected = known
    else:
        selected = [x.strip() for x in requested.split(",") if x.strip()]
    unknown = [x for x in selected if x not in CASE_BY_ID]
    if unknown:
        raise ValueError(f"unknown case(s): {unknown}; choose from {known}")

    if args.include_optional:
        for cid in ("numba_umap", "torch_cpu"):
            if cid not in selected:
                selected.append(cid)
    if args.include_gpu and "torch_cuda" not in selected:
        selected.append("torch_cuda")

    skipped = []
    available = []
    for cid in selected:
        cap = CASE_BY_ID[cid].required_capability
        if cap and not caps.get(cap, False):
            skipped.append({"case_id": cid, "reason": f"capability unavailable: {cap}"})
        else:
            available.append(cid)
    return available, skipped


def _case_config(args: argparse.Namespace, case_id: str) -> dict[str, Any]:
    return {
        "threads": int(args.threads),
        "linalg_size": int(args.linalg_size),
        "linalg_repeats": 3,
        "rf_samples": 3000, "rf_features": 24, "rf_trees": 80,
        "opt_evals": int(args.opt_evals),
        "umap_samples": 3000,
        "torch_samples": 1024, "torch_features": 32, "torch_epochs": 5,
    }


def _case_env(base: dict[str, str], args: argparse.Namespace,
              case_id: str) -> dict[str, str]:
    env = dict(base)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(PROJECT_ROOT), env.get("PYTHONPATH", "")
    ]))
    # Keep all common CPU pools bounded unless the user explicitly requests a count.
    for key in THREAD_ENV:
        env[key] = str(args.threads)
    if case_id == "cpu_rf":
        env["MLTB_NJOBS"] = str(args.n_jobs)
    if case_id == "numba_umap":
        env["NUMBA_NUM_THREADS"] = str(args.numba_threads)
    if case_id in ("torch_cpu", "torch_cuda"):
        env["CUDA_VISIBLE_DEVICES"] = ("" if case_id == "torch_cpu" else
                                       str(args.cuda_device))
    return env


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort tree cleanup after a case timeout."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=15, check=False)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    cols = ["case_id", "status", "n", "wall_p50_s", "wall_p95_s", "cpu_p50_s",
            "rss_peak_max_mb", "gpu_allocated_peak_max_mb",
            "gpu_reserved_peak_max_mb", "consistent", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in results:
            st = rec.get("stats", {})
            cons = rec.get("consistency", {})
            writer.writerow({
                "case_id": rec.get("case_id", ""), "status": rec.get("status", ""),
                "n": st.get("wall_s", {}).get("n", 0),
                "wall_p50_s": st.get("wall_s", {}).get("p50"),
                "wall_p95_s": st.get("wall_s", {}).get("p95"),
                "cpu_p50_s": st.get("cpu_s", {}).get("p50"),
                "rss_peak_max_mb": st.get("rss_peak_mb", {}).get("max"),
                "gpu_allocated_peak_max_mb":
                    st.get("gpu_allocated_peak_mb", {}).get("max"),
                "gpu_reserved_peak_max_mb":
                    st.get("gpu_reserved_peak_mb", {}).get("max"),
                "consistent": cons.get("stable"),
                "error": rec.get("error", ""),
            })


def run_parent(args: argparse.Namespace) -> int:
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        raise ValueError("warmup must be >=0; repeats and threads must be >=1")
    caps = capabilities()
    selected, skipped = _resolve_cases(args.cases, caps, args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = build_manifest(args)
    out = Path(args.output).expanduser().resolve() if args.output else \
        (PERF_ROOT / "results" / f"{stamp}-pid{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    manifest["selected_cases"] = selected
    manifest["skipped_cases"] = skipped
    _atomic_json(out / "manifest.json", manifest)

    results = []
    for idx, case_id in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] {case_id} ...", flush=True)
        case_file = out / f"{idx:02d}-{case_id}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", case_id, "--worker-output", str(case_file),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        env = _case_env(os.environ, args, case_id)
        env["PERF_CONFIG"] = json.dumps(_case_config(args, case_id))
        proc = None
        try:
            # Do not capture child stdio: this avoids Windows named-pipe restrictions
            # in restricted runners. The child writes a structured result file.
            proc = subprocess.Popen(cmd, env=env,
                                    start_new_session=(os.name != "nt"))
            returncode = proc.wait(timeout=float(args.case_timeout))
            if returncode != 0 and not case_file.exists():
                rec = {"schema": 1, "status": "error", "case_id": case_id,
                       "error": f"worker exited {returncode}"}
                _atomic_json(case_file, rec)
            rec = json.loads(case_file.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            if proc is not None:
                _terminate_process_tree(proc)
            rec = {"schema": 1, "status": "error", "case_id": case_id,
                   "error": f"worker timeout after {args.case_timeout}s"}
            _atomic_json(case_file, rec)
        except Exception as exc:
            rec = {"schema": 1, "status": "error", "case_id": case_id,
                   "error": f"{type(exc).__name__}: {exc}"}
            _atomic_json(case_file, rec)
        results.append(rec)
        wall = rec.get("stats", {}).get("wall_s", {})
        cons = rec.get("consistency", {})
        print(f"  {rec.get('status')}: P50={wall.get('p50')}s "
              f"P95={wall.get('p95')}s consistent={cons.get('stable')}", flush=True)

    manifest["case_input_sha256"] = {
        rec.get("case_id", ""): rec.get("input_sha256")
        for rec in results if rec.get("case_id")
    }
    _atomic_json(out / "manifest.json", manifest)
    summary = {
        "schema": 1, "manifest": "manifest.json", "skipped": skipped,
        "cases": results,
    }
    _atomic_json(out / "summary.json", summary)
    _write_summary_csv(out / "summary.csv", results)
    print(f"Results: {out}")
    return 1 if any(r.get("status") != "ok" for r in results) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list cases and exit")
    p.add_argument("--cases", default="quick",
                   help="quick | all | comma-separated case ids")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=1,
                   help="OMP/MKL/BLAS/torch CPU thread count")
    p.add_argument("--n-jobs", default="1", help="MLTB_NJOBS for cpu_rf")
    p.add_argument("--numba-threads", type=int, default=1)
    p.add_argument("--include-optional", action="store_true")
    p.add_argument("--include-gpu", action="store_true")
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument("--linalg-size", type=int, default=384)
    p.add_argument("--opt-evals", type=int, default=24)
    p.add_argument("--case-timeout", type=float, default=600.0)
    p.add_argument("--output", default="", help="explicit output directory")
    p.add_argument("--worker", choices=sorted(CASE_BY_ID), help=argparse.SUPPRESS)
    p.add_argument("--worker-output", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for spec in CASE_SPECS:
            req = spec.required_capability or "core"
            default = "default" if spec.default else "optional"
            print(f"{spec.case_id:14s} [{default:7s}; {req:10s}] {spec.title}")
        return 0
    if args.worker:
        if not args.worker_output:
            parser.error("--worker-output is required with --worker")
        try:
            config = json.loads(os.environ.get("PERF_CONFIG", "{}"))
        except json.JSONDecodeError:
            config = {}
        run_worker(args.worker, Path(args.worker_output), config,
                   int(args.warmup), int(args.repeats))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
