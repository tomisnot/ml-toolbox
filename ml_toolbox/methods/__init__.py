# -*- coding: utf-8 -*-
"""内置方法族。导入本包即完成注册。

neural 子包（torch）单独 try：未装 torch 的环境其余 52 方法照常可用。
"""
from . import linear, ensemble, svm, knn, bayes, cluster, manifold, anomaly, timeseries, baseline, quantum  # noqa: F401
try:
    from . import neural  # noqa: F401
except Exception:
    pass
