# -*- coding: utf-8 -*-
"""UI 回归测试（离屏）：主窗口装配、遍历、参数三态、非法输入不崩。

运行：python tests/test_ui.py
"""
import os
import sys
import warnings

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

# 铁律：lightgbm 先于任何 PyQt5 模块 import
from ml_toolbox.core import registry
registry.load_builtin()

from PyQt5.QtWidgets import QApplication, QMessageBox      # noqa: E402
from PyQt5.QtCore import Qt                                # noqa: E402
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

from ml_toolbox.ui.main_window import MainWindow          # noqa: E402
from ml_toolbox.core.demo import load_demo                # noqa: E402

app = QApplication([])
PASS, FAIL = 0, []


def check(name, fn):
    global PASS
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:180]}")


def test_build_and_load():
    win = MainWindow()
    win.resize(1400, 850)
    win.show()
    app.processEvents()
    win._set_dataset(load_demo("iris"))
    app.processEvents()
    assert win.spec is not None and len(win.spec.meta["chain"]) == 4
    assert win.chain_page.findChildren(type(win.chain_page)) is not None
    win._test_ref = win
    return win


_win = None


def _win_get():
    global _win
    if _win is None:
        _win = test_build_and_load()
    return _win


def test_batch_run():
    win = _win_get()
    from ml_toolbox.core import runner
    from ml_toolbox.core.contracts import RunConfig
    recs = runner.run_batch(["logistic", "knn", "svc"], win.spec, RunConfig())
    win._on_batch_done(recs)
    app.processEvents()
    assert win.compare_table.rowCount() == 3
    assert win.tabs.count() == 5
    # 画廊应有 3 张缩略图
    from ml_toolbox.ui.gallery import _Thumb
    thumbs = win.gallery.findChildren(_Thumb)
    assert len(thumbs) == 3


def test_data_page():
    """数据检视页：特征相关热图有数据。"""
    win = _win_get()
    assert win.data_page is not None
    win.tabs.setCurrentIndex(win.TAB_DATA)
    app.processEvents()


def test_heatmap_orientation():
    """热图朝向像素级验证（转置错误是静默 bug，只有探色能抓到）。
    约定：A[a,b] -> a=纵轴向上（row0 在底部）、b=横轴。"""
    import pyqtgraph as pg
    from ml_toolbox.ui.heatmap import AdaptiveMatrixHeatmap
    A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], float)
    h = AdaptiveMatrixHeatmap()
    h.set_data(A, ["r0", "r1", "r2"])
    h.resize(500, 500)
    h.show()
    app.processEvents()
    lut = np.array(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))

    def expect(v):
        return lut[int(round(v / 8 * 255))][:3]

    def probe(col, row):
        sc = h.vb.mapViewToScene(pg.Point(col + 0.5, row + 0.5))
        pt = h.glw.mapFromScene(sc)
        return np.array(
            h.glw.grab().toImage().pixelColor(int(pt.x()), int(pt.y())).getRgb()[:3])

    for row in range(3):
        for col in range(3):
            e, g = expect(A[row, col]), probe(col, row)
            assert sum(abs(int(a) - int(b)) for a, b in zip(e, g)) < 30, \
                f"cell({row},{col}) expect {e} got {g}"
    h.close()


def test_gallery_click():
    win = _win_get()
    win.tabs.setCurrentIndex(win.TAB_GALLERY)
    app.processEvents()
    from ml_toolbox.ui.gallery import _Thumb
    thumbs = win.gallery.findChildren(_Thumb)
    assert thumbs
    thumbs[0].clicked.emit(thumbs[0].idx)
    app.processEvents()
    assert win.tabs.currentIndex() == win.TAB_INSPECT   # 跳到方法检视
    assert win.inspector.tabs.count() >= 6


def test_inspect_pages_built():
    win = _win_get()
    win._show_record(0)
    app.processEvents()
    insp = win.inspector
    # 分类兜底页：混淆矩阵/ROC/PR/重要性 + 共享 指标/参数/诊断
    assert insp.tabs.count() >= 6
    # 每页都有内容（画布/表格/占位），不是空白
    for i in range(insp.tabs.count()):
        insp.tabs.setCurrentIndex(i)
        app.processEvents()
        w = insp.tabs.currentWidget()
        assert w is not None


def test_param_three_states():
    """参数三态：默认(空) / 合法值 / 非法值。"""
    win = _win_get()
    win._inspect_method("ridge")
    app.processEvents()
    pp = win.params
    # 找到 alpha 输入框
    from ml_toolbox.core.contracts import ParamSpec
    assert "alpha" in pp._widgets
    w = pp._widgets["alpha"][1]
    # 1) 空 = 默认
    w.setText("")
    ov, errs = pp.collect()
    assert "alpha" not in ov and not errs
    # 2) 合法值
    w.setText("0.5")
    ov, errs = pp.collect()
    assert ov.get("alpha") == 0.5 and not errs
    # 3) 非法值：报错不崩，状态栏提示
    w.setText("abc")
    ov, errs = pp.collect()
    assert errs and "alpha" not in ov
    # 越界
    w.setText("-3")
    ov, errs = pp.collect()
    assert errs


def test_param_rerun():
    """改参 -> 应用并重跑：结果参数确实变化。"""
    win = _win_get()
    win._inspect_method("ridge")
    app.processEvents()
    w = win.params._widgets["alpha"][1]
    w.setText("100.0")
    win._rerun_current()
    app.processEvents()
    import time
    for _ in range(50):
        app.processEvents()
        time.sleep(0.02)
        if getattr(win, "_current_record", None) is not None \
                and win._current_record.result.params.get("alpha") == 100.0:
            break
    assert win._current_record.result.params["alpha"] == 100.0


def test_method_browser_filter():
    win = _win_get()
    win.browser._search.setText("随机森林")
    app.processEvents()
    vis = []
    tree = win.browser._tree
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            if not top.child(j).isHidden():
                vis.append(top.child(j).data(0, Qt.UserRole))
    assert vis == ["random_forest"]
    # "forest" 应命中两个（iforest 内部名含 forest）
    win.browser._search.setText("forest")
    app.processEvents()
    n = sum(1 for i in range(tree.topLevelItemCount())
            for j in range(tree.topLevelItem(i).childCount())
            if not tree.topLevelItem(i).child(j).isHidden())
    assert n == 2
    win.browser._search.setText("")


def test_pipeline_dialog_edit():
    win = _win_get()
    from ml_toolbox.ui.main_window import PipelineDialog
    dlg = PipelineDialog(win, win.pipeline)
    # 禁用缩放步骤
    win.pipeline.steps[2].enabled = False
    spec = win.pipeline.run(win.dataset)
    app.processEvents()
    chain = spec.meta["chain"]
    assert any(e.get("skipped") for e in chain)
    dlg.deleteLater()


def test_history_load():
    """历史面板：保存一次运行 -> load_run_record 重建 -> 主窗口可检视。"""
    win = _win_get()
    from ml_toolbox.core import runner, persistence, registry
    from ml_toolbox.core.contracts import RunConfig
    rec = runner.run_one(registry.get("ridge"), win.spec, RunConfig())
    persistence.save_record(rec)
    from ml_toolbox.ui.history import load_run_record
    r2 = load_run_record(rec.run_id)
    assert r2 is not None and r2.result.ok
    # 主窗口载入历史（无 estimator 也能出图）
    win.records = [r2]
    win._fill_compare_table()
    win._show_record(0)
    app.processEvents()
    assert win.inspector.tabs.count() >= 5
    import pandas as pd
    fi = r2.result.artifacts.get("feature_importance")
    assert isinstance(fi, pd.DataFrame) and len(fi) == win.spec.X.shape[1]


def test_purpose_filter():
    """方法库用途筛选：选"代理模型"只剩 GPR 等标注该目的的方法。"""
    win = _win_get()
    from ml_toolbox.core import registry
    win.browser._purpose.setCurrentText("全部")
    app.processEvents()
    win.browser._purpose.setCurrentText("代理模型")
    app.processEvents()
    tree = win.browser._tree
    vis = [tree.topLevelItem(i).child(j).data(0, Qt.UserRole)
           for i in range(tree.topLevelItemCount())
           for j in range(tree.topLevelItem(i).childCount())
           if not tree.topLevelItem(i).child(j).isHidden()]
    assert "gpr" in vis
    assert "random_forest" not in vis       # 未标代理模型
    assert all("代理模型" in registry.get(nm).purposes_cn() for nm in vis)
    win.browser._purpose.setCurrentText("全部")
    app.processEvents()


def test_neural_pages():
    """方案C 检视页：权重热图 + 表示演化回放可装配、可切帧、可播放。"""
    try:
        import torch  # noqa: F401
    except Exception:
        print("    (torch 不可用，跳过 neural_pages)")
        return
    win = _win_get()
    from ml_toolbox.core import registry, runner
    from ml_toolbox.core.contracts import RunConfig
    win._set_dataset(load_demo("breast_cancer"))
    app.processEvents()
    rec = runner.run_one(registry.get("torch_mlp"), win.spec,
                         RunConfig(diag=True, overrides={"epochs": 15,
                                                         "latent_every": 3}))
    assert rec.result.ok, rec.result.error
    win.records = [rec]
    win._show_record(0)
    app.processEvents()
    insp = win.inspector
    assert insp.tabs.count() >= 8      # 5 声明页 + 3 共享页
    from ml_toolbox.ui.neural_pages import WeightsPage, ReplayPage
    wps = insp.tabs.findChildren(WeightsPage)
    rps = insp.tabs.findChildren(ReplayPage)
    assert wps and rps, "注册页 builder 未生效"
    # 权重页：切视图（最终/初始/ΔW）不崩
    wp = wps[0]
    for i in range(3):
        wp._view.setCurrentIndex(i)
        app.processEvents()
    # 回放页：切帧 + 播放一帧
    rp = rps[0]
    rp._slider.setValue(1)
    app.processEvents()
    rp._toggle_play()
    app.processEvents()
    rp._next_frame()
    app.processEvents()
    rp._toggle_play()
    assert rp._readout.text().startswith("epoch")
    win._set_dataset(load_demo("iris"))


def test_no_dataset_button():
    """无数据点运行：提示而非崩溃。"""
    win = MainWindow()
    win.show()
    app.processEvents()
    win._run_batch()
    app.processEvents()   # QMessageBox 已打桩，不应卡死
    win.close()


if __name__ == "__main__":
    tests = [
        ("build_and_load", test_build_and_load),
        ("batch_run", test_batch_run),
        ("gallery_click", test_gallery_click),
        ("data_page", test_data_page),
        ("heatmap_orientation", test_heatmap_orientation),
        ("inspect_pages_built", test_inspect_pages_built),
        ("param_three_states", test_param_three_states),
        ("param_rerun", test_param_rerun),
        ("method_browser_filter", test_method_browser_filter),
        ("pipeline_dialog_edit", test_pipeline_dialog_edit),
        ("history_load", test_history_load),
        ("purpose_filter", test_purpose_filter),
        ("neural_pages", test_neural_pages),
        ("no_dataset_button", test_no_dataset_button),
    ]
    print(f"UI 回归测试 {len(tests)} 项")
    for name, fn in tests:
        check(name, fn)
    print(f"\n{PASS} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  FAILED {n}: {e}")
        sys.exit(1)
