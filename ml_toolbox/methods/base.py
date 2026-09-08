# -*- coding: utf-8 -*-
"""sklearn 风格监督学习统一封装：子类只声明 estimator 工厂与参数 schema。

fit/predict/evaluate/交叉验证/特征重要性全部在基类完成（方法间资源共享）。
双任务方法（target_kind=None）按数据自动选择 reg_factory / cls_factory。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import MLMethod, RunConfig, TASK_SUPERVISED
from ..core.pipeline import _infer_kind


class SklearnSupervised(MLMethod):
    """task=supervised 的 sklearn estimator 通用壳。"""

    target_kind = None  # 'regression' | 'classification' | None(双任务)

    def fit_extra_artifacts(self, X, result) -> dict:
        """基类默认无附加产物；监督子类覆写。"""
        return {}

    def make(self, params: dict, kind: str):
        """子类实现：kind in ('regression','classification')。"""
        raise NotImplementedError

    # ------------------------------------------------ 统一实现
    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        params = self.params(cfg)
        kind = self.target_kind or _infer_kind(y)
        est = self.make(params, kind)
        Xv = X.to_numpy(float)
        self._le = None
        if kind == "classification":
            from sklearn.preprocessing import LabelEncoder
            self._le = LabelEncoder()
            yv = self._le.fit_transform(np.asarray(y))
        else:
            yv = np.asarray(y, float)
        est.fit(Xv, yv)
        res = self._new_result(target_kind=kind, params=params)
        res.est = est
        # 标签编码器随 result 走：换一个新方法实例也能 predict（独立引用友好）。
        # 存"原始类别"（_le.classes_），est.classes_ 是编码后的 0..k-1，不能用。
        res.classes_ = (self._le.classes_ if kind == "classification"
                        and self._le is not None else None)
        fi = _feature_importance(est, list(X.columns))
        if fi is None and diag:
            # 无原生重要性：仅诊断模式下补 permutation（默认零开销 —— P3）
            fi = _permutation_importance(est, Xv, yv, list(X.columns), kind)
        if fi is not None:
            res.artifacts["feature_importance"] = fi
        if diag:
            res.diag = {"n_train": int(Xv.shape[0]),
                        "n_features": int(Xv.shape[1]),
                        "estimator": repr(est)[:500]}
        return res

    def predict(self, X: pd.DataFrame, result):
        est = getattr(result, "est", None)
        if est is None:
            raise RuntimeError("结果中缺少已拟合 estimator（可能是加载的历史记录）")
        Xv = X.to_numpy(float)
        if result.target_kind == "classification":
            labels = est.predict(Xv)
            le = getattr(self, "_le", None)
            if le is None:
                # 实例未拟合过（换新实例调用）：从 result 携带的信息重建编码器
                import sklearn.preprocessing as spp
                classes = getattr(result, "classes_", None)
                if classes is None:
                    return labels
                le = spp.LabelEncoder()
                le.classes_ = np.asarray(classes)
                self._le = le
            return le.inverse_transform(labels)
        return est.predict(Xv)

    def fit_extra_artifacts(self, X: pd.DataFrame, result) -> dict:
        """run_one 在 predict 前调用：概率等附加产物（非监督方法默认无）。"""
        est = getattr(result, "est", None)
        out = {}
        if est is not None and result.target_kind == "classification" \
                and hasattr(est, "predict_proba"):
            try:
                out["y_prob"] = est.predict_proba(X.to_numpy(float))
            except Exception:
                pass
        return out


def _feature_importance(est, cols=None):
    """统一特征重要性提取：importance / 系数绝对值。"""
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, float)
    elif hasattr(est, "coef_"):
        c = np.asarray(est.coef_, float)
        imp = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)
    else:
        return None
    idx = cols if cols is not None and len(cols) == len(imp) \
        else [f"f{i}" for i in range(len(imp))]
    return pd.DataFrame({"importance": imp}, index=idx)


def _permutation_importance(est, Xv, yv, cols, kind):
    """无 coef_/feature_importances_ 的方法（如 SVR/KNN）用置换重要性兜底。
    仅在 diag=True 时调用（计算量大，默认零开销）。"""
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(est, Xv, yv, n_repeats=5,
                                   random_state=42, n_jobs=1)
        imp = np.asarray(r.importances_mean, float)
        idx = cols if len(cols) == len(imp) else [f"f{i}" for i in range(len(imp))]
        return pd.DataFrame({"importance": imp}, index=idx)
    except Exception:
        return None
