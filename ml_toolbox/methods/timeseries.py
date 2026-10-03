# -*- coding: utf-8 -*-
"""时序预测族（常见非神经网络方法）。

契约：fit 接收按时间排序的 y 序列；cfg.extras['horizon'] 指定预测步数
（默认 = 后 20% 作为保留测试段）。核心图 = 历史 + 预测 + 置信区间。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import (MLMethod, RunConfig, PageSpec, ParamSpec,
                              TASK_TIMESERIES)
from ..core.registry import register
from . import plots


class TSMethod(MLMethod):
    task = TASK_TIMESERIES
    family = "timeseries"
    primary = "rmse"

    #: **是否消费外生特征**（`problem.X` 的列）。
    #: ⚠ 诚实声明，别乱开：8 个时序方法里**只有 ARIMA/SARIMAX 数学上支持外生变量**；
    #: SES/Holt/HoltWinters/Theta/朴素法都是单变量方法。不支持而数据带了外生列时，
    #: 基类会把这件事**写进 diag**（`unused_features`）——不静默忽略
    #: （老契约下"时序忽略外生 X"就是静默忽略，见 README 记的已知限制）。
    supports_exog: bool = False

    def forecast(self, y_train: np.ndarray, horizon: int, p: dict, exog=None):
        """子类实现：返回 (fcst, ci95) 两个长度为 horizon 的数组。

        ``exog``：``(exog_train, exog_future)`` 或 ``None``（只有 `supports_exog`
        的方法会收到非 None，见 `fit`）。
        """
        raise NotImplementedError

    def fit(self, problem, cfg: RunConfig, diag: bool = False):
        """时序任务的 fit（`Problem` 范式）。

        ⚠ 语义与迁移前**逐字一致**：原来 runner 传进来的是 `spec.split()` 的**训练段**，
        所以这里取 `problem.rows("train")`；时间切分仍由本族内部按 `horizon` 做
        （= 训练段的后 horizon 个点做保留测试）。这样既有的基准结论不受影响。
        """
        X, y = problem.rows("train")
        p = self.params(cfg)
        s = np.asarray(pd.Series(y).to_numpy(float), float)
        n = len(s)
        h = int(cfg.extras.get("horizon", max(2, int(n * 0.2))))
        h = max(1, min(h, n - 4))
        tr, te = s[: n - h], s[n - h:]

        # ── 外生特征：**只有声明支持的方法才收**，其余如实披露（不静默）
        feat_cols = [c for c in getattr(X, "columns", [])]
        exog = None
        unused = []
        if feat_cols:
            if self.supports_exog and n > h:
                M = np.asarray(X.to_numpy(float), float)
                if len(M) == n:
                    exog = (M[: n - h], M[n - h:])
                else:
                    unused = feat_cols          # 行数对不上 ⇒ 不敢用，如实说
            else:
                unused = feat_cols

        fcst, ci = self.forecast(tr, h, p, exog=exog)
        fcst = np.asarray(fcst, float).ravel()
        ci = np.asarray(ci, float).ravel() if ci is not None else None
        res = self._new_result(target_kind="regression", params=p)
        res.artifacts = {
            "t_true": np.arange(n, dtype=float),
            "y_true": s,
            "forecast": np.concatenate([tr, fcst]).astype(float),
            "ci": np.concatenate([np.zeros(len(tr)), ci]) if ci is not None
                  else None,
            "y_test": te, "f_test": fcst, "horizon": h,
        }
        from ..core.contracts import regression_metrics
        res.metrics.update(regression_metrics(te, fcst))
        res.primary_metric = "rmse"
        if diag or unused:
            # ⚠ `unused` 时**即使没开 diag 也写**：'数据带了外生列却没被用'属于
            #   必须让人看见的事实（静默忽略正是要防的假绿）。
            res.diag = {"n_train": int(len(tr)), "horizon": h,
                        "test_rmse": res.metrics.get("rmse")}
            if unused:
                res.diag["unused_features"] = list(unused)
                res.diag["unused_reason"] = (
                    f"本方法（{self.name}）不支持外生变量"
                    if not self.supports_exog else "外生列行数与序列对不齐")
            if exog is not None:
                res.diag["exog_used"] = int(np.asarray(exog[0]).shape[1])
        return res

    def inspect_pages(self, cfg):
        return [PageSpec("fc", "预测曲线", "mpl", plots.plot_forecast),
                PageSpec("res", "残差序列", "mpl", plots.plot_ts_residual)]


@register
class NaiveLast(TSMethod):
    name = "ts_naive"
    display_name = "朴素法（末值/漂移）"
    tags = ("baseline", "must-beat")
    param_schema = [
        ParamSpec("drift", "允许漂移", "bool", True,
                  hint="末值 + 平均趋势 × 步数"),
    ]

    def forecast(self, tr, h, p, exog=None):
        last = tr[-1]
        if bool(p["drift"]) and len(tr) > 2:
            slope = (tr[-1] - tr[0]) / (len(tr) - 1)
            fc = last + slope * np.arange(1, h + 1)
        else:
            fc = np.full(h, last)
        sd = float(np.std(np.diff(tr))) if len(tr) > 2 else 0.0
        ci = 1.96 * sd * np.sqrt(np.arange(1, h + 1))
        return fc, ci


@register
class LinearTrend(TSMethod):
    name = "ts_trend"
    display_name = "线性趋势外推"
    tags = ("trend", "interpretable")
    param_schema = [
        ParamSpec("damped", "阻尼系数", "number", 1.0, min=0.5, max=1.0,
                  hint="<1 抑制远期外推"),
    ]

    def forecast(self, tr, h, p, exog=None):
        t = np.arange(len(tr), dtype=float)
        coef = np.polyfit(t, tr, 1)
        te = np.arange(len(tr), len(tr) + h, dtype=float)
        fc = np.polyval(coef, te)
        d = float(p["damped"])
        if d < 1:
            base = tr[-1]
            fc = base + (fc - base) * d * np.arange(1, h + 1) / h
        resid = tr - np.polyval(coef, t)
        sd = float(np.std(resid))
        return fc, 1.96 * sd * np.sqrt(1 + np.arange(h) / max(len(tr), 1))


@register
class SES(TSMethod):
    name = "ses"
    display_name = "简单指数平滑"
    tags = ("smoothing", "no-trend")
    param_schema = [
        ParamSpec("alpha", "平滑系数 α", "number", 0.3, min=0.02, max=1.0),
    ]

    def forecast(self, tr, h, p, exog=None):
        a = float(p["alpha"])
        level = tr[0]
        errs = []
        for v in tr[1:]:
            errs.append(v - level)
            level = a * v + (1 - a) * level
        sd = float(np.std(errs)) if errs else 0.0
        return np.full(h, level), 1.96 * sd * np.ones(h)


@register
class Holt(TSMethod):
    name = "holt"
    display_name = "Holt 双参数指数平滑"
    tags = ("smoothing", "trend")
    param_schema = [
        ParamSpec("alpha", "水平 α", "number", 0.4, min=0.02, max=1.0),
        ParamSpec("beta", "趋势 β", "number", 0.1, min=0.01, max=1.0),
        ParamSpec("damped", "阻尼趋势", "bool", True),
    ]

    def forecast(self, tr, h, p, exog=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(tr, trend="add",
                                 damped_trend=bool(p["damped"]),
                                 seasonal=None,
                                 initialization_method="estimated")
        fit = m.fit(smoothing_level=float(p["alpha"]),
                    smoothing_trend=float(p["beta"]))
        fc = np.asarray(fit.forecast(h), float)
        resid = np.asarray(tr - fit.fittedvalues[~np.isnan(fit.fittedvalues)], float)
        sd = float(np.std(resid)) if len(resid) else 0.0
        return fc, 1.96 * sd * np.sqrt(1 + np.arange(h) / max(len(tr), 1))


@register
class HoltWinters(TSMethod):
    name = "holt_winters"
    display_name = "Holt-Winters 三参数（季节性）"
    tags = ("smoothing", "seasonal")
    param_schema = [
        ParamSpec("seasonal_period", "季节周期", "int", 12, min=2,
                  hint="如月度数据年内周期=12、周=7"),
        ParamSpec("seasonal", "季节模式", "select", "additive",
                  choices=["additive", "multiplicative"]),
    ]

    def forecast(self, tr, h, p, exog=None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        per = int(p["seasonal_period"])
        if len(tr) < 2 * per:
            per = max(2, len(tr) // 2)
        m = ExponentialSmoothing(tr, trend="add", seasonal=p["seasonal"],
                                 seasonal_periods=per,
                                 initialization_method="estimated")
        fit = m.fit(optimized=True)
        fc = np.asarray(fit.forecast(h), float)
        fitted = np.asarray(fit.fittedvalues, float)
        resid = tr - fitted
        sd = float(np.nanstd(resid))
        return fc, 1.96 * sd * np.sqrt(1 + np.arange(h) / max(len(tr), 1))


@register
class ARIMA(TSMethod):
    name = "arima"
    display_name = "ARIMA / SARIMA"
    tags = ("statistical", "classical", "seasonal")
    #: SARIMAX 支持外生回归量 ⇒ 它**真的**消费 `problem.X`（8 个时序方法里唯一一个）。
    supports_exog = True
    param_schema = [
        ParamSpec("p", "AR 阶 p", "int", 1, min=0, max=5),
        ParamSpec("d", "差分阶 d", "int", 1, min=0, max=2),
        ParamSpec("q", "MA 阶 q", "int", 1, min=0, max=5),
        ParamSpec("seasonal_period", "季节周期（0=无）", "int", 0, min=0),
        ParamSpec("P", "季节 AR", "int", 1, min=0, max=2),
        ParamSpec("D", "季节差分", "int", 0, min=0, max=1),
        ParamSpec("Q", "季节 MA", "int", 1, min=0, max=2),
    ]

    def forecast(self, tr, h, p, exog=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = (int(p["p"]), int(p["d"]), int(p["q"]))
        per = int(p["seasonal_period"])
        seasonal = ((int(p["P"]), int(p["D"]), int(p["Q"]), per)
                    if per >= 2 and len(tr) >= 2 * per else None)
        # ⭐ **唯一真支持外生变量的时序方法**（SARIMAX 带 exog）。
        #   未来期的外生值必须一起给（预测 h 步要用到），所以基类把
        #   `(exog_train, exog_future)` 成对传进来；缺一半就退回单变量
        #   （并由基类在 diag 里披露），**不做"只喂训练段"的半截支持**（那会静默错）。
        ex_tr = ex_fu = None
        if exog is not None:
            ex_tr, ex_fu = exog
            ex_tr = np.asarray(ex_tr, float)
            ex_fu = np.asarray(ex_fu, float)
            if len(ex_tr) != len(tr) or len(ex_fu) != h:
                ex_tr = ex_fu = None
        m = SARIMAX(tr, exog=ex_tr, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
        fit = m.fit(disp=False, maxiter=200)
        if ex_fu is not None:
            got = fit.get_forecast(h, exog=ex_fu)
            fc = np.asarray(got.predicted_mean, float)
            se = np.asarray(got.se_mean, float)
            return fc, 1.96 * np.nan_to_num(se)
        fc = np.asarray(fit.forecast(h), float)
        se = np.asarray(fit.forecast(h).se if hasattr(fit.forecast(h), "se")
                        else np.full(h, np.nanstd(tr)), float)
        return fc, 1.96 * np.nan_to_num(se)


@register
class Theta(TSMethod):
    name = "theta"
    display_name = "Theta 模型"
    tags = ("classical", "competition")
    param_schema = []

    def forecast(self, tr, h, p, exog=None):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        per = max(2, len(tr) // 4)
        try:
            m = ThetaModel(pd.Series(tr), period=per).fit()
            fc = np.asarray(m.forecast(h), float)
            ci = m.conf_int()
            width = np.asarray(ci.iloc[:, 1] - ci.iloc[:, 0], float) / 2 \
                if ci is not None and len(ci) >= h else None
            return fc, (1.96 * np.nan_to_num(width) if width is not None else None)
        except Exception:
            return NaiveLast.forecast(self, tr, h, {"drift": True})
