# -*- coding: utf-8 -*-
"""集成学习族：bagging / boosting / stacking 入口。"""
from __future__ import annotations

# ⚠ LightGBM 必须在 QApplication 构造之前完成 import，否则其 OpenMP 运行时
# 与 Qt 冲突 -> fit 时 access violation（实测 import 顺序是唯一可靠解法）。
# 顶层导入即保证：任何 import 本模块的进程都会先加载 lightgbm。
try:
    import lightgbm as _lgb  # noqa: F401
except Exception:            # 未安装/环境异常时降级，LightGBM 方法运行时才报错
    _lgb = None

# ⚠ 并行度单一控制点 = core.parallel（M5/C5）：成员模型 n_jobs 走 nj()（默认 -1），
# 元学习器（Voting/Stacking）外层走 meta_nj()（默认 1——成员已并行，外层再并行
# 会嵌套超订，Windows OpenMP 线程池互相挤兑，曾致 stacking 偶发 access violation）。
# 受限环境设 MLTB_NJOBS=off（全串行 + 钉 numba），见 core/parallel.py。
from ..core.parallel import nj, meta_nj, restricted

from ..core.contracts import ParamSpec
from ..core.registry import register
from .base import SklearnSupervised


@register
class RandomForest(SklearnSupervised):
    name = "random_forest"
    display_name = "随机森林"
    family = "ensemble"
    target_kind = None
    tags = ("bagging", "importance", "robust")
    requires_numeric = False
    param_schema = [
        ParamSpec("n_estimators", "树数量", "int", 300, min=10),
        ParamSpec("max_depth", "最大深度", "int", 0, min=0,
                  hint="0 = 不限制"),
        ParamSpec("min_samples_leaf", "叶最小样本", "int", 1, min=1),
        ParamSpec("class_weight", "类别加权", "select", "none",
                  choices=["none", "balanced_subsample"]),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        kw = dict(n_estimators=int(p["n_estimators"]), random_state=42,
                  n_jobs=nj(),
                  max_depth=int(p["max_depth"]) or None,
                  min_samples_leaf=int(p["min_samples_leaf"]))
        if kind == "classification" and p["class_weight"] != "none":
            kw["class_weight"] = p["class_weight"]
        return (RandomForestClassifier(**kw) if kind == "classification"
                else RandomForestRegressor(**kw))


@register
class ExtraTrees(SklearnSupervised):
    name = "extra_trees"
    display_name = "极端随机树"
    family = "ensemble"
    target_kind = None
    tags = ("bagging", "importance", "fast")
    param_schema = [
        ParamSpec("n_estimators", "树数量", "int", 300, min=10),
        ParamSpec("max_depth", "最大深度", "int", 0, min=0),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        kw = dict(n_estimators=int(p["n_estimators"]), random_state=42,
                  n_jobs=nj(),
                  max_depth=int(p["max_depth"]) or None)
        return (ExtraTreesClassifier(**kw) if kind == "classification"
                else ExtraTreesRegressor(**kw))


@register
class GradientBoosting(SklearnSupervised):
    name = "gradient_boosting"
    display_name = "梯度提升树 (GBDT)"
    family = "ensemble"
    target_kind = None
    tags = ("boosting", "tabular-strong")
    param_schema = [
        ParamSpec("n_estimators", "迭代轮数", "int", 200, min=10),
        ParamSpec("learning_rate", "学习率", "number", 0.1, min=0.005, max=1),
        ParamSpec("max_depth", "树深度", "int", 3, min=1, max=10),
        ParamSpec("subsample", "行采样", "number", 1.0, min=0.3, max=1),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import (GradientBoostingRegressor,
                                      GradientBoostingClassifier)
        kw = dict(n_estimators=int(p["n_estimators"]),
                  learning_rate=float(p["learning_rate"]),
                  max_depth=int(p["max_depth"]),
                  subsample=float(p["subsample"]), random_state=42)
        return (GradientBoostingClassifier(**kw) if kind == "classification"
                else GradientBoostingRegressor(**kw))


@register
class HistGB(SklearnSupervised):
    name = "hist_gb"
    display_name = "直方图梯度提升"
    family = "ensemble"
    target_kind = None
    tags = ("boosting", "fast", "tabular-strong")
    param_schema = [
        ParamSpec("max_iter", "迭代轮数", "int", 300, min=10),
        ParamSpec("learning_rate", "学习率", "number", 0.1, min=0.005, max=1),
        ParamSpec("max_depth", "最大深度", "int", 0, min=0),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        kw = dict(max_iter=int(p["max_iter"]),
                  learning_rate=float(p["learning_rate"]),
                  max_depth=int(p["max_depth"]) or None, random_state=42)
        return (HistGradientBoostingClassifier(**kw) if kind == "classification"
                else HistGradientBoostingRegressor(**kw))


@register
class AdaBoost(SklearnSupervised):
    name = "adaboost"
    display_name = "AdaBoost"
    family = "ensemble"
    target_kind = None
    tags = ("boosting",)
    param_schema = [
        ParamSpec("n_estimators", "弱学习器数", "int", 100, min=10),
        ParamSpec("learning_rate", "学习率", "number", 1.0, min=0.01),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
        kw = dict(n_estimators=int(p["n_estimators"]),
                  learning_rate=float(p["learning_rate"]), random_state=42)
        return (AdaBoostClassifier(**kw) if kind == "classification"
                else AdaBoostRegressor(**kw))


@register
class XGBoost(SklearnSupervised):
    name = "xgboost"
    display_name = "XGBoost"
    family = "ensemble"
    target_kind = None
    tags = ("boosting", "competition", "tabular-strong")
    param_schema = [
        ParamSpec("n_estimators", "迭代轮数", "int", 300, min=10),
        ParamSpec("max_depth", "树深度", "int", 6, min=1, max=16),
        ParamSpec("learning_rate", "学习率", "number", 0.1, min=0.005, max=1),
        ParamSpec("subsample", "行采样", "number", 0.9, min=0.3, max=1),
        ParamSpec("colsample_bytree", "列采样", "number", 0.9, min=0.3, max=1),
        ParamSpec("reg_lambda", "L2 正则", "number", 1.0, min=0),
    ]

    def make(self, p, kind):
        import xgboost as xgb
        kw = dict(n_estimators=int(p["n_estimators"]),
                  max_depth=int(p["max_depth"]),
                  learning_rate=float(p["learning_rate"]),
                  subsample=float(p["subsample"]),
                  colsample_bytree=float(p["colsample_bytree"]),
                  reg_lambda=float(p["reg_lambda"]),
                  random_state=42, n_jobs=nj(), verbosity=0)
        if kind == "classification":
            return xgb.XGBClassifier(**kw)
        return xgb.XGBRegressor(**kw)


@register
class LightGBM(SklearnSupervised):
    name = "lightgbm"
    display_name = "LightGBM"
    family = "ensemble"
    target_kind = None
    tags = ("boosting", "fast", "tabular-strong")
    param_schema = [
        ParamSpec("n_estimators", "迭代轮数", "int", 300, min=10),
        ParamSpec("num_leaves", "叶数", "int", 31, min=2),
        ParamSpec("learning_rate", "学习率", "number", 0.1, min=0.005, max=1),
        ParamSpec("min_child_samples", "叶最小样本", "int", 20, min=1),
    ]

    def make(self, p, kind):
        import lightgbm as lgb
        kw = dict(n_estimators=int(p["n_estimators"]),
                  num_leaves=int(p["num_leaves"]),
                  learning_rate=float(p["learning_rate"]),
                  min_child_samples=int(p["min_child_samples"]),
                  random_state=42, verbose=-1)
        # 注意：LightGBM 的 OpenMP 多线程在 Windows 子线程（QThread）中
        # 会触发 access violation，故不并行（sklearn/xgboost 无此问题）。
        # 例外：MLTB_NJOBS=off 的受限环境里，loky 的核数子进程探测本身就会
        # PermissionError——此时显式 n_jobs=1 跳过 joblib 路径（C5）。
        if restricted():
            kw["n_jobs"] = 1
        if kind == "classification":
            return lgb.LGBMClassifier(**kw)
        return lgb.LGBMRegressor(**kw)


@register
class Voting(SklearnSupervised):
    name = "voting"
    display_name = "投票/平均集成"
    family = "ensemble"
    target_kind = None
    tags = ("meta",)
    param_schema = [
        ParamSpec("voting", "方式", "select", "soft", choices=["soft", "hard"]),
        ParamSpec("members", "成员", "text", "random_forest,xgboost,extra_trees",
                  hint="逗号分隔的已注册方法名（须支持当前任务类型）"),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from ..core import registry
        from ..core.contracts import RunConfig
        registry.load_builtin()
        names = [s.strip() for s in str(p["members"]).split(",") if s.strip()]
        ests = []
        for nm in names:
            m = registry.get(nm)
            if m.task != "supervised":
                continue
            # 成员必须支持当前任务类型（回归/分类）
            if m.target_kind is not None and m.target_kind != kind:
                continue
            ests.append((nm, m.make(m.params(RunConfig()), kind)))
        if len(ests) < 2:
            raise ValueError(f"有效成员不足 2 个（kind={kind}，成员={names}）")
        if kind == "classification":
            return VotingClassifier(ests, voting=p["voting"], n_jobs=meta_nj())
        return VotingRegressor(ests, n_jobs=meta_nj())


@register
class Stacking(SklearnSupervised):
    name = "stacking"
    display_name = "堆叠集成"
    family = "ensemble"
    target_kind = None
    tags = ("meta", "stacking")
    param_schema = [
        ParamSpec("members", "基学习器", "text",
                  "random_forest,extra_trees,gradient_boosting",
                  hint="逗号分隔的已注册方法名（须支持当前任务类型）"),
        ParamSpec("final_alpha", "元模型正则", "number", 1.0, min=1e-4,
                  hint="Ridge 元模型的 alpha"),
    ]

    def make(self, p, kind):
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import Ridge, LogisticRegression
        from ..core import registry
        from ..core.contracts import RunConfig
        registry.load_builtin()
        names = [s.strip() for s in str(p["members"]).split(",") if s.strip()]
        ests = []
        for nm in names:
            m = registry.get(nm)
            if m.task != "supervised" or (m.target_kind is not None
                                          and m.target_kind != kind):
                continue
            ests.append((nm, m.make(m.params(RunConfig()), kind)))
        if len(ests) < 2:
            raise ValueError(f"基学习器不足 2 个（kind={kind}）")
        if kind == "classification":
            final = LogisticRegression(max_iter=1000, C=float(p["final_alpha"]))
            return StackingClassifier(ests, final_estimator=final, cv=4,
                                      n_jobs=meta_nj())
        final = Ridge(alpha=float(p["final_alpha"]))
        return StackingRegressor(ests, final_estimator=final, cv=4,
                                 n_jobs=meta_nj())
