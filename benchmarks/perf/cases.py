# -*- coding: utf-8 -*-
"""Local algorithm cases for the minimal performance harness.

Heavy dependencies are imported inside ``prepare`` so ``--list`` and unit tests stay
light.  Every case is a fresh object in a fresh Python process; measured ``run``
calls create fresh estimators/models and return a deterministic consistency payload.

No external AI service, model API, network call, GUI, or business persistence is
used by this module.
"""
from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    required_capability: str | None = None
    default: bool = False
    tags: tuple[str, ...] = ()


CASE_SPECS = (
    CaseSpec("cpu_linalg", "NumPy matrix multiply", default=True,
             tags=("cpu", "blas", "quick")),
    CaseSpec("opt_random", "Random-search optimization", default=True,
             tags=("cpu", "optimization", "quick")),
    CaseSpec("cpu_rf", "Toolbox RandomForest", tags=("cpu", "joblib", "algorithm")),
    CaseSpec("numba_umap", "Local UMAP/Numba", "umap",
             tags=("cpu", "numba", "optional")),
    CaseSpec("torch_cpu", "Existing local torch_mlp on CPU", "torch",
             tags=("cpu", "torch", "optional")),
    CaseSpec("torch_cuda", "Existing local torch_mlp on CUDA", "torch_cuda",
             tags=("gpu", "cuda", "optional")),
)

CASE_BY_ID = {spec.case_id: spec for spec in CASE_SPECS}


def _array_sha256(*arrays) -> str:
    """Stable hash for immutable benchmark inputs, computed outside timing."""
    h = hashlib.sha256()
    for arr in arrays:
        a = arr.tobytes(order="C") if hasattr(arr, "tobytes") else bytes(arr)
        dtype = str(getattr(arr, "dtype", type(arr).__name__))
        shape = tuple(getattr(arr, "shape", (len(arr),)))
        h.update(dtype.encode("utf-8"))
        h.update(str(shape).encode("ascii"))
        h.update(a)
    return h.hexdigest()


class PerfCase(abc.ABC):
    """Prepare once outside timing, then run a fresh computation per repeat."""

    case_id = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    def prepare(self) -> None:
        """Build immutable inputs and warm imports outside measured time."""

    @abc.abstractmethod
    def run(self) -> Any:
        """Return a JSON-serializable consistency payload."""

    def close(self) -> None:
        """Optional per-process cleanup."""

    def input_sha256(self) -> str | None:
        """Hash immutable case inputs; computed during prepare, never timed."""
        return None


class NumpyLinalgCase(PerfCase):
    case_id = "cpu_linalg"

    def prepare(self) -> None:
        import numpy as np

        n = int(self.config.get("linalg_size", 384))
        repeats = int(self.config.get("linalg_repeats", 3))
        rng = np.random.default_rng(20260718)
        self.a = rng.standard_normal((n, n), dtype=np.float64)
        self.b = rng.standard_normal((n, n), dtype=np.float64)
        self.repeats = repeats

    def input_sha256(self) -> str:
        return _array_sha256(self.a, self.b)

    def run(self) -> Any:
        import numpy as np

        out = self.a @ self.b
        for _ in range(self.repeats - 1):
            out = self.a @ out
        finite = bool(np.isfinite(out).all())
        return {
            "checksum": float(np.sum(out, dtype=np.float64)),
            "trace": [float(x) for x in np.diag(out)[:8]],
            "finite": finite,
            "shape": list(out.shape),
        }


class RandomForestCase(PerfCase):
    case_id = "cpu_rf"

    def prepare(self) -> None:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification

        from ml_toolbox.core.pipeline import Pipeline
        from ml_toolbox.core.dataset import Dataset

        n = int(self.config.get("rf_samples", 3000))
        d = int(self.config.get("rf_features", 24))
        trees = int(self.config.get("rf_trees", 80))
        n_inf = min(12, max(1, d - 1))
        n_red = min(4, max(0, d - n_inf - 1))
        X, y = make_classification(
            n_samples=n, n_features=d, n_informative=n_inf,
            n_redundant=n_red, n_classes=3, n_clusters_per_class=1,
            class_sep=0.9, random_state=42,
        )
        frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
        frame["target"] = y.astype(int)
        ds = Dataset(frame, name="perf_cpu_rf", target="target")
        self.spec = Pipeline.default().run(ds)
        self.cfg = {"overrides": {"n_estimators": trees}, "seed": 42}

    def input_sha256(self) -> str:
        return _array_sha256(self.spec.X.to_numpy(float),
                             self.spec.y.to_numpy())

    def run(self) -> Any:
        from ml_toolbox.core import registry, runner
        from ml_toolbox.core.contracts import RunConfig

        registry.load_builtin()
        rec = runner.run_one(registry.get("random_forest"), self.spec,
                             RunConfig(**self.cfg))
        if not rec.result.ok:
            raise RuntimeError(rec.result.error or "random_forest failed")
        return {
            "y_pred": rec.result.artifacts["y_pred"],
            "metrics": rec.result.metrics,
            "params": rec.result.params,
        }


class RandomOptimizationCase(PerfCase):
    case_id = "opt_random"

    def prepare(self) -> None:
        from ml_toolbox.core.contracts import ParamSpec
        from ml_toolbox.opt.contracts import ParamSpace

        self.space = ParamSpace([
            ParamSpec("x1", "x1", "number", 0.0, min=-2.0, max=2.0),
            ParamSpec("x2", "x2", "number", 0.0, min=-2.0, max=2.0),
            ParamSpec("x3", "x3", "number", 0.0, min=-2.0, max=2.0),
        ])

        def six_hump(p):
            x, y = p["x1"], p["x2"]
            return float((4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2
                         + x * y + (-4 + 4 * y ** 2) * y ** 2
                         + 0.1 * p["x3"] ** 2)

        from ml_toolbox.opt.contracts import make_objective
        self.objective = make_objective(six_hump, self.space, name="perf_six_hump")
        self.n_evals = int(self.config.get("opt_evals", 24))

    def input_sha256(self) -> str:
        payload = {"space": self.space.describe(), "objective": "perf_six_hump",
                   "seed": 42, "n_evals": self.n_evals}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def run(self) -> Any:
        from ml_toolbox.opt import registry
        from ml_toolbox.opt.contracts import Budget
        from ml_toolbox.opt.runner import optimize

        registry.load_builtin()
        rec = optimize(self.objective, registry.get("random_search"),
                       Budget(n_evals=self.n_evals), seed=42)
        if rec.error:
            raise RuntimeError(rec.error)
        history = rec.history.drop(columns=["ts"], errors="ignore")
        return {
            "history": history.to_dict(orient="list"),
            "best": rec.best,
        }


class UmapCase(PerfCase):
    case_id = "numba_umap"

    def prepare(self) -> None:
        import numpy as np
        from sklearn.datasets import load_digits

        data = load_digits()
        self.X = data.data[: int(self.config.get("umap_samples", 3000))]
        self.n_neighbors = int(self.config.get("umap_neighbors", 15))
        self.min_dist = float(self.config.get("umap_min_dist", 0.1))

    def input_sha256(self) -> str:
        return _array_sha256(np.asarray(self.X, dtype=np.float64))

    def run(self) -> Any:
        import numpy as np
        from sklearn.manifold import trustworthiness
        from umap import UMAP

        estimator = UMAP(n_components=2, n_neighbors=self.n_neighbors,
                         min_dist=self.min_dist, random_state=42)
        emb = np.asarray(estimator.fit_transform(self.X), dtype=np.float64)
        return {
            "embedding": emb,
            "trustworthiness": float(trustworthiness(
                self.X, emb, n_neighbors=min(15, len(self.X) - 1))),
            "finite": bool(np.isfinite(emb).all()),
        }


class TorchMlpCase(PerfCase):
    case_id = "torch_cpu"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.device = "cuda" if self.case_id == "torch_cuda" else "cpu"

    def prepare(self) -> None:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.datasets import make_classification

        from ml_toolbox.core.pipeline import Pipeline
        from ml_toolbox.core.dataset import Dataset

        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            torch.cuda.empty_cache()

        n = int(self.config.get("torch_samples", 1024))
        d = int(self.config.get("torch_features", 32))
        n_inf = min(16, max(1, d - 1))
        n_red = min(4, max(0, d - n_inf - 1))
        X, y = make_classification(
            n_samples=n, n_features=d, n_informative=n_inf,
            n_redundant=n_red, n_classes=3, n_clusters_per_class=1,
            random_state=42,
        )
        frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
        frame["target"] = y.astype(int)
        ds = Dataset(frame, name=f"perf_{self.case_id}", target="target")
        self.spec = Pipeline.default().run(ds)
        self.epochs = int(self.config.get("torch_epochs", 5))
        self.torch = torch

    def input_sha256(self) -> str:
        return _array_sha256(self.spec.X.to_numpy(float),
                             self.spec.y.to_numpy())

    def run(self) -> Any:
        from ml_toolbox.core import registry, runner
        from ml_toolbox.core.contracts import RunConfig

        registry.load_builtin()
        cfg = RunConfig(overrides={
            "device": self.device,
            "hidden": "32,16",
            "epochs": self.epochs,
            "batch": 64,
            "latent_every": 100,
        }, seed=42)
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.synchronize()
        rec = runner.run_one(registry.get("torch_mlp"), self.spec, cfg)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        if not rec.result.ok:
            raise RuntimeError(rec.result.error or "torch_mlp failed")
        payload = {
            "y_pred": rec.result.artifacts["y_pred"],
            "metrics": rec.result.metrics,
            "params": rec.result.params,
            "device": self.device,
        }
        if self.device == "cuda":
            payload["gpu_allocated_peak_bytes"] = int(
                self.torch.cuda.max_memory_allocated())
            payload["gpu_reserved_peak_bytes"] = int(
                self.torch.cuda.max_memory_reserved())
        return payload


class TorchCudaCase(TorchMlpCase):
    case_id = "torch_cuda"


CASE_TYPES = {
    "cpu_linalg": NumpyLinalgCase,
    "opt_random": RandomOptimizationCase,
    "cpu_rf": RandomForestCase,
    "numba_umap": UmapCase,
    "torch_cpu": TorchMlpCase,
    "torch_cuda": TorchCudaCase,
}


def make_case(case_id: str, config: dict[str, Any] | None = None) -> PerfCase:
    try:
        cls = CASE_TYPES[case_id]
    except KeyError as exc:
        raise KeyError(f"unknown perf case: {case_id}") from exc
    return cls(config or {})
