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

    def forecast(self, y_train: np.ndarray, horizon: int, p: dict):
        """子类实现：返回 (fcst, ci95) 两个长度为 horizon 的数组。"""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y, cfg: RunConfig, diag: bool = False):
        p = self.params(cfg)
        s = np.asarray(pd.Series(y).to_numpy(float), float)
        n = len(s)
        h = int(cfg.extras.get("horizon", max(2, int(n * 0.2))))
        h = max(1, min(h, n - 4))
        tr, te = s[: n - h], s[n - h:]
        fcst, ci = self.forecast(tr, h, p)
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
        if diag:
            res.diag = {"n_train": int(len(tr)), "horizon": h,
                        "test_rmse": res.metrics.get("rmse")}
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

    def forecast(self, tr, h, p):
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

    def forecast(self, tr, h, p):
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

    def forecast(self, tr, h, p):
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

    def forecast(self, tr, h, p):
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

    def forecast(self, tr, h, p):
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
    param_schema = [
        ParamSpec("p", "AR 阶 p", "int", 1, min=0, max=5),
        ParamSpec("d", "差分阶 d", "int", 1, min=0, max=2),
        ParamSpec("q", "MA 阶 q", "int", 1, min=0, max=5),
        ParamSpec("seasonal_period", "季节周期（0=无）", "int", 0, min=0),
        ParamSpec("P", "季节 AR", "int", 1, min=0, max=2),
        ParamSpec("D", "季节差分", "int", 0, min=0, max=1),
        ParamSpec("Q", "季节 MA", "int", 1, min=0, max=2),
    ]

    def forecast(self, tr, h, p):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = (int(p["p"]), int(p["d"]), int(p["q"]))
        per = int(p["seasonal_period"])
        seasonal = ((int(p["P"]), int(p["D"]), int(p["Q"]), per)
                    if per >= 2 and len(tr) >= 2 * per else None)
        m = SARIMAX(tr, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
        fit = m.fit(disp=False, maxiter=200)
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

    def forecast(self, tr, h, p):
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
