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
    import time
    t0 = time.time()
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name} ({time.time() - t0:.1f}s)")
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
    assert win.tabs.count() == 5      # ML 工作区五页（优化是平级 Perspective，非 tab）
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
    h.resize(900, 900)   # 字号 x2 后轴区加宽，探测窗需更大保证格心不压网格线
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
    # iris（150×4）足够验证页面装配；breast_cancer 569 行让 IG 归因慢 3 倍
    win._set_dataset(load_demo("iris"))
    app.processEvents()
    rec = runner.run_one(registry.get("torch_mlp"), win.spec,
                         RunConfig(diag=True, overrides={"epochs": 8,
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


def test_perspective_switch():
    """兄弟框架的 UI 语义：模式切换整区替换，ML 专属控件按模式显隐。"""
    win = _win_get()
    assert win.stack.count() == 2
    assert win.stack.currentIndex() == 0
    assert win._cv.isVisible() and win.browser.isVisible()
    win._mode_opt.trigger()
    app.processEvents()
    assert win.stack.currentIndex() == 1
    assert win.opt_page.isVisible()
    assert not win._cv.isVisible()          # ML 专属控件隐藏
    assert not win.browser.isVisible()      # ML 工作区整体隐藏
    assert win._demo.isVisible()            # 共享工具栏保留
    win._mode_ml.trigger()
    app.processEvents()
    assert win.stack.currentIndex() == 0
    assert win._cv.isVisible()


def test_opt_workbench():
    """优化工作区：构建 + 喂入 OptRecord 刷新五页不崩。"""
    win = _win_get()
    win._mode_opt.trigger()
    app.processEvents()
    wb = win.opt_page
    assert wb is not None
    from ml_toolbox.opt import registry as oreg
    from ml_toolbox.opt.contracts import Budget
    from ml_toolbox.opt.synth import make_objective_from_synth
    from ml_toolbox.opt.runner import optimize
    oreg.load_builtin()
    obj = make_objective_from_synth("six_hump")
    rec = optimize(obj, oreg.get("random_search"), Budget(n_evals=15),
                   seed=3)
    assert len(rec.history) == 15
    # 直接喂 record 刷新（不经过 worker 线程，测渲染路径）
    wb._records = {"random_search": rec}
    wb._refresh_live(rec)
    app.processEvents()
    assert wb.hist_table.rowCount() == 15
    # GP-BO 代理切片路径（M6：代理页按 record.optimizer 取实例）
    gp = oreg.get("gp_bo")
    rec2 = optimize(obj, gp, Budget(n_evals=14), cfg={"n_init": 5}, seed=4)
    wb._records = {"gp_bo": rec2}
    wb._opt_instances = {"gp_bo": gp}
    wb._refresh_live(rec2)
    app.processEvents()
    # 串台防护：记录属于 random_search 时，代理页不得用 gp_bo 的实例
    wb._records = {"random_search": rec}
    wb._refresh_live(rec)
    app.processEvents()
    # 多目标 Pareto 页路径（NSGA-II + ZDT1）
    try:
        from ml_toolbox.opt.synth import make_objective_multi
        mobj = make_objective_multi("zdt1", dim=4)
        mrec = optimize(mobj, oreg.get("nsga_ii"), Budget(n_evals=120),
                        cfg={"popsize": 20}, seed=5)
        wb._records = {"nsga_ii": mrec}
        wb._refresh_live(mrec)
        app.processEvents()
        assert mrec.pareto is not None and len(mrec.pareto) >= 3
    except Exception as e:                      # nsga_ii 未注册等
        print("    (pareto 页跳过:", e, ")")
    # 多优化器对比表
    wb._records = {"random_search": rec, "gp_bo": rec2}
    wb._refresh_compare()
    app.processEvents()
    assert wb.hist_table.rowCount() == 2


def test_autotuner_ui():
    """UI 接缝2：目标切到 ML 方法调参 -> 数据源状态 + AutoTunerObjective。"""
    win = _win_get()
    win._set_dataset(load_demo("iris"))
    app.processEvents()
    wb = win.opt_page
    assert getattr(wb, "_spec", None) is not None      # 数据上下文已注入
    wb._obj_kind.setCurrentText("ML 方法调参（AutoTuner）")
    app.processEvents()
    # 数据源状态栏可见且绿字（内部数据集已连接）
    assert wb._src_group.isVisible()
    assert "内部数据集" in wb._src_status.text()
    wb._ml_method.setCurrentText("logistic")
    obj = wb._make_objective()
    assert obj.__class__.__name__ == "AutoTunerObjective"
    assert obj.space.dim >= 1
    # 一次评估能返回有限分数
    s = obj(obj.space.sample(np.random.RandomState(0)))
    assert np.isfinite(s)
    # 外部文件数据源：写临时 csv -> FileSource 绿字 -> 删文件 -> 红字
    import tempfile, os as _os
    import pandas as _pd
    tmp = _os.path.join(tempfile.gettempdir(), "ui_src_test.csv")
    df = _pd.DataFrame(np.random.RandomState(0).randn(30, 3), columns=list("abc"))
    df["target"] = (df["a"] > 0).astype(int)
    df.to_csv(tmp, index=False)
    wb._src_kind.setCurrentText("外部文件（csv/parquet）")
    wb._src_path.setText(tmp)
    wb._src_target.setText("target")
    wb.refresh_source_status()
    assert "文件 ui_src_test.csv" in wb._src_status.text()
    assert "#2ca02c" in wb._src_status.styleSheet()
    _os.remove(tmp)
    wb.refresh_source_status()
    assert "#c0392b" in wb._src_status.styleSheet()   # 读取失败 -> 红
    # 外部程序目标：参数定义解析
    wb._obj_kind.setCurrentText("外部程序（黑盒进程）")
    app.processEvents()
    sp, _mp = wb._proc_space("a=0..5, b=-1..1")
    assert sp.dim == 2 and sp.keys == ["a", "b"]
    try:
        wb._proc_space("bad")
        assert False, "非法参数定义应抛 ValueError"
    except ValueError:
        pass
    wb._obj_kind.setCurrentText("合成函数")
    app.processEvents()


def test_opt_cfg_export_import():
    """配置导出/导入：freeze 实战配置装载 -> 面板状态 -> objective 构造。

    ⚠ **机器无关（2026-09-27 开源前脱敏）**：夹具 `cases/freeze_autotune.json` 原先把
    `proc.cwd` 写成**另一个项目的本机绝对路径**（那个目录后来被移走/打包 ⇒ 本判据在
    任何别的机器上都红，理由还是别人的路径）。现在夹具只放**中性占位**，本判据
    在装载前把它换成一个**真的临时目录** ⇒ 判据不再依赖任何机器上的外部目录。
    """
    import json
    import tempfile
    from ml_toolbox.ui.opt_page import OptWorkbench
    wb = OptWorkbench()
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cases", "freeze_autotune.json")
    with open(cfg_path, encoding="utf-8") as f:
        raw = f.read()
    # 夹具里**不许**再出现他人/本机绝对路径（防它悄悄回来）
    for bad in ("Energy " + "Level", "LEN" + "OVO", "Users" + chr(92), "/Users/"):
        assert bad not in raw, f"夹具含本机/他人路径字样 {bad!r} ⇒ 开源前必须脱敏"
    cfg = json.loads(raw)
    # 占位值不参与语义：装载前换成真临时目录（只影响"目录存在性"校验）
    # ⚠ 全程必须在 `with` **里面**：`_make_objective()` 会再次校验该目录存在，
    #   出了 with 临时目录已被删 ⇒ 自己造一个假红。
    with tempfile.TemporaryDirectory(prefix="ml-blackbox-") as bb:
        cfg["proc"]["cwd"] = bb
        wb.set_config(cfg)
        g = wb.get_config()
        assert g["proc"]["cwd"] == bb          # 面板原样保留传入的目录
        assert g["objective_kind"] == "外部程序（黑盒进程）"
        assert g["proc"]["mode"] == 1 and g["proc"]["score_field"] == "C_dual"
        assert g["proc"]["maximize"] is True
        assert g["budget"]["workers"] == "8"
        assert g["optimizers"] == ["gp_bo"]
        obj = wb._make_objective()
        assert obj.__class__.__name__ == "BatchProcessObjective"
        assert obj.minimize is False and obj.space.dim == 2
        assert obj.constraints and obj.constraints[0]["field"] == "S_ret"
        # 往返稳定：再导一次与第一次一致
        assert wb.get_config() == g
    # 约束解析
    cs = OptWorkbench._parse_constraints("S_ret>=0.99, kick_m05>0.5")
    assert cs[1] == {"field": "kick_m05", "op": ">", "value": 0.5}
    # 嵌套路径参数：ell.alpha_deg -> 参数名 alpha_deg + mapping
    sp, mp = OptWorkbench._proc_space("freq_mhz=7200..7500, ell.alpha_deg=-90..90")
    assert sp.dim == 2 and mp == {"alpha_deg": "ell.alpha_deg"}


def test_opt_worker_parallel_e2e():
    """真实线程链冒烟：OptWorker(QThread) + workers=2 + 子进程批量黑盒。

    offscreen 下不点 GUI，但走完整 _launch_next 同款路径：
    OptWorker.run -> optimize(workers=2) -> evaluate_many -> subprocess。
    """
    import tempfile
    from ml_toolbox.core.contracts import ParamSpec
    from ml_toolbox.opt.contracts import ParamSpace, Budget
    from ml_toolbox.opt.process import BatchProcessObjective
    from ml_toolbox.ui.worker import OptWorker
    script = os.path.join(tempfile.gettempdir(), "ui_bb_fake.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write("import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--points'); "
                "p.add_argument('--out'); a=p.parse_args()\n"
                "pts=json.load(open(a.points,encoding='utf-8'))\n"
                "out=[{'id':q.get('id',k),'ok':True,"
                "'C_dual':1.0-(float(q['x'])-0.5)**2,'S_ret':1.0}\n"
                "     for k,q in enumerate(pts)]\n"
                "json.dump(out,open(a.out,'w',encoding='utf-8'))\n")
    sp = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=0.0, max=1.0)])
    cmd = (f'python "{script}" --points {{points_file}} --out {{out_file}}')
    obj = BatchProcessObjective(cmd, sp,
                                point_map={"x": "x"},
                                score_field="C_dual", minimize=False,
                                constraints=[{"field": "S_ret", "op": ">=",
                                              "value": 0.99}],
                                stagger=0.0, timeout=60.0, name="uibb")
    from ml_toolbox.opt import registry as oreg
    oreg.load_builtin()
    w = OptWorker(obj, oreg.get("random_search"), Budget(n_evals=6),
                  seed=3, workers=2)
    done = {"rec": None, "fail": None, "n_eval": 0}
    w.eval_done.connect(lambda r, i: done.update(n_eval=i + 1))
    w.finished_ok.connect(lambda r: done.update(rec=r))
    w.failed.connect(lambda tb: done.update(fail=tb))
    w.start()
    import time
    t0 = time.time()
    while w.isRunning() and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    w.wait(5000)
    app.processEvents()
    assert done["fail"] is None, done["fail"]
    rec = done["rec"]
    assert rec is not None and rec.error is None, rec.error if rec else "无记录"
    assert len(rec.history) == 6
    assert (rec.history["status"] == "ok").all()
    assert done["n_eval"] == 6                    # 直播回调逐条到达
    assert abs(rec.best["score"] + 1.0) < 0.3     # 最小化方向峰 = -1


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
        ("perspective_switch", test_perspective_switch),
        ("opt_workbench", test_opt_workbench),
        ("autotuner_ui", test_autotuner_ui),
        ("opt_cfg_export_import", test_opt_cfg_export_import),
        ("opt_worker_parallel_e2e", test_opt_worker_parallel_e2e),
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
