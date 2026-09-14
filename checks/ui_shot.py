# -*- coding: utf-8 -*-
"""离屏 UI 截图工作流（UI design/SKILL.md 五步闭环的第②步）。

用法：python checks/ui_shot.py  ->  生成 checks/shots/_ui_shot_*.png  ->  AI 视觉自查。
截图是临时检查产物，统一落在 checks/shots/（已 gitignore），不散落根目录。

铁律落实（勿删）：
1. CJK 字体显式加载（offscreen 无中文 -> 方框）；
2. 导入顺序：lightgbm 先于任何 PyQt5 模块 import（OpenMP 冲突，见 methods/ensemble.py）；
3. matplotlib grab 前同步 draw()（MplCanvas 用 draw_idle 排程，
   截图前统一 canvas.draw()）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

# ⚠ 导入顺序铁律：lightgbm（经 registry.load_builtin）必须先于任何 PyQt5
#   模块 import，否则 OpenMP 与 Qt 冲突 -> fit 时 access violation。
from ml_toolbox.core import registry                       # noqa: E402
registry.load_builtin()                                     # noqa: E402

from PyQt5.QtWidgets import QApplication, QMessageBox      # noqa: E402

QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.critical = staticmethod(lambda *a, **k: None)

from ml_toolbox.ui.main_window import MainWindow          # noqa: E402
from ml_toolbox.core.demo import load_demo                # noqa: E402

FULL_W, FULL_H = 1920, 1080

app = QApplication([])
from PyQt5.QtGui import QFontDatabase, QFont              # noqa: E402
for _fn in ("msyh.ttc", "msyh.ttf", "simhei.ttf", "simsun.ttc"):
    _p = os.path.join("C:\\", "Windows", "Fonts", _fn)
    if os.path.exists(_p):
        _fid = QFontDatabase.addApplicationFont(_p)
        _fams = QFontDatabase.applicationFontFamilies(_fid)
        if _fams:
            app.setFont(QFont(_fams[0], 10))
            print("UI 字体:", _fams[0])
            break

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
# 截图产物统一目录（临时检查产物，勿散落根目录）
SHOTS = os.path.join(ROOT, "checks", "shots")
os.makedirs(SHOTS, exist_ok=True)


def shot_path(name):
    return os.path.join(SHOTS, f"_ui_shot_{name}.png")


def build_window():
    win = MainWindow()
    win.resize(FULL_W, FULL_H)
    win.show()
    app.processEvents()
    return win


def load_scenario(win):
    """真实数据填充：iris 演示集 + 选一批代表性方法（开启 CV 验证列显示）。"""
    win._set_dataset(load_demo("iris"))
    app.processEvents()
    names = ["logistic", "random_forest", "svc", "knn", "gaussian_nb",
             "xgboost", "lightgbm", "gradient_boosting"]
    from ml_toolbox.core import runner
    from ml_toolbox.core.contracts import RunConfig
    cfg = RunConfig(diag=True, extras={"cv_folds": 5})
    recs = runner.run_batch(names, win.spec, cfg)
    win._on_batch_done(recs)
    app.processEvents()
    return recs


# 逐场景验证"每个方法核心图不一样"（G2）：数据集 -> 方法 -> 该族专属图
SCENARIOS = [
    ("housing_small", "lightgbm", "reg"),       # 回归：拟合散点+残差+重要性
    ("blobs", "kmeans", "clu"),                 # 聚类：散点+轮廓
    ("timeseries_synth", "holt_winters", "ts"),  # 时序：预测曲线+置信区间
    ("anomaly_synth", "iforest", "ano"),        # 异常：分数直方图+阈值
    ("digits", "pca", "man"),                   # 降维：嵌入散点+方差解释
    ("breast_cancer", "torch_mlp", "nn"),       # 神经网络：方案C 五页检视
]


def capture_scenarios(win):
    """每个场景单独设数据+跑方法+截检视页，验证不同族渲染不同核心图。
    注意：场景切换会重建检视页（旧页 deleteLater），故必须即时 grab。"""
    from ml_toolbox.ui.widgets import MplCanvas
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    shots = []
    for demo, method, tag in SCENARIOS:
        try:
            win._set_dataset(load_demo(demo))
            app.processEvents()
            m = registry.get(method)
            rec = runner.run_one(m, win.spec, RunConfig(diag=True))
            win.records = [rec]
            win._show_record(0)
            app.processEvents()
            win.tabs.setCurrentIndex(4)
            insp = win.inspector
            n = insp.tabs.count()
            for i in range(n):
                insp.tabs.setCurrentIndex(i)
                app.processEvents()
                for c in insp.tabs.currentWidget().findChildren(MplCanvas):
                    c.draw()
                app.processEvents()
                pix = insp.tabs.currentWidget().grab()   # 即时抓取
                shots.append((f"{tag}_{method}_{i}", pix))
        except Exception as e:
            print(f"场景 {demo}/{method} 跳过: {e}")
    return shots


def capture_list(win):
    """返回 [(name, callable)]；每个 callable 自己切换状态再截图
    （列表构建与截图执行之间隔着主循环，状态必须在 callable 内设置）。"""
    from ml_toolbox.ui.widgets import MplCanvas
    out = []

    def shot(name, prep):
        def _go():
            prep()
            app.processEvents()
            return win.grab() if name in ("chain", "data", "compare",
                                          "inspect", "gallery") \
                else win.inspector.tabs.currentWidget().grab()
        out.append((name, _go))

    # tab 布局：0处理链 1数据检视 2对比 3画廊 4检视
    shot("chain", lambda: win.tabs.setCurrentIndex(0))
    shot("data", lambda: win.tabs.setCurrentIndex(1))
    shot("compare", lambda: win.tabs.setCurrentIndex(2))

    def _prep_gallery():
        win.tabs.setCurrentIndex(3)
        app.processEvents()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        for c in win.gallery.findChildren(FigureCanvasQTAgg):
            c.draw()
    shot("gallery", _prep_gallery)
    shot("inspect", lambda: win.tabs.setCurrentIndex(4))

    insp = win.inspector
    for i in range(insp.tabs.count()):
        def prep(i=i):
            win.tabs.setCurrentIndex(4)
            insp.tabs.setCurrentIndex(i)
            app.processEvents()
            for c in insp.tabs.currentWidget().findChildren(MplCanvas):
                c.draw()
        shot(f"inspect_page_{i}", prep)
    return out


def capture_opt(win):
    """优化工作区：喂入两条 OptRecord 渲染四页 + 对比表（不依赖线程时序）。"""
    from ml_toolbox.opt import registry as oreg
    from ml_toolbox.opt.contracts import Budget
    from ml_toolbox.opt.synth import make_objective_from_synth
    from ml_toolbox.opt.runner import optimize
    from ml_toolbox.ui.widgets import MplCanvas
    shots = []
    try:
        oreg.load_builtin()
        win._mode_opt.trigger()              # Perspective：切到优化工作区
        app.processEvents()
        wb = win.opt_page
        obj = make_objective_from_synth("ackley_3d")
        gp = oreg.get("gp_bo")
        r1 = optimize(obj, gp, Budget(n_evals=25), cfg={"n_init": 6}, seed=7)
        r2 = optimize(obj, oreg.get("random_search"), Budget(n_evals=25),
                      seed=7)
        wb._opt_instances = {"gp_bo": gp}   # C4-1：代理页按 record.optimizer 取实例
        wb._records = {"gp_bo": r1, "random_search": r2}
        wb._refresh_live(r1)
        wb._refresh_compare()
        app.processEvents()
        tabs = wb.inspector.tabs            # C4-1：检视页改由声明式装配器持有
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            for c in tabs.currentWidget().findChildren(MplCanvas):
                c.draw()
            app.processEvents()
            shots.append((f"opt_page_{i}", tabs.currentWidget().grab()))
        shots.append(("opt_full", win.grab()))
        # 多目标 Pareto 页
        from ml_toolbox.opt.synth import make_objective_multi
        mobj = make_objective_multi("zdt1", dim=6)
        mrec = optimize(mobj, oreg.get("nsga_ii"), Budget(n_evals=1200),
                        cfg={"popsize": 20}, seed=9)
        wb._records = {"nsga_ii": mrec}
        wb._refresh_live(mrec)
        app.processEvents()
        pi = [i for i, p in enumerate(wb.inspector._pages)
              if p.spec.key == "pareto"]
        if pi:
            wb.inspector.tabs.setCurrentIndex(pi[0])   # 按 key 定位（页序声明化）
            app.processEvents()
            for c in wb.inspector.tabs.currentWidget().findChildren(MplCanvas):
                c.draw()
            app.processEvents()
            shots.append(("opt_pareto", wb.inspector.tabs.currentWidget().grab()))
    except Exception as e:
        print(f"优化工作区截图跳过: {e}")
    return shots


if __name__ == "__main__":
    win = build_window()
    load_scenario(win)
    app.processEvents()
    for name, grab_callable in capture_list(win):
        app.processEvents()
        grab_callable().save(shot_path(name))
        print("截图 " + shot_path(name))
    # 逐场景（不同方法族核心图）
    win._set_dataset(load_demo("iris"))   # 复位再跑场景，避免污染
    for name, pix in capture_scenarios(win):
        pix.save(shot_path(name))
        print("截图 " + shot_path(name))
    # 优化工作区
    for name, pix in capture_opt(win):
        pix.save(shot_path(name))
        print("截图 " + shot_path(name))
    win.close()
    print("完成 —— 用视觉模态读取 PNG，按 checklist.md 自查。")
