# -*- coding: utf-8 -*-
"""主窗口：两种对等工作模式（Perspective）——ML 方法试验台 / 优化调参。

UI 层次映射内部框架层次（重设计）：
  顶栏      品牌 + 胶囊模式开关（兄弟框架的顶层状态切换）+ 数据工具
            （打开/演示/管道/历史 共享；诊断/CV/种子/预测 ML 专属，按模式显隐）
  ML 模式   左=方法库（注册表视图）| 中=阶段流导航 + 内容栈（数模工作流：
            看数据→选模型→看细节→调参）| 右=参数面板（就地重跑）
  优化模式   OptWorkbench 全权接管（四段流水线卡片 + 检视/历史）

兄弟框架在 UI 上的语义：模式平级、共享数据工具栏；ML 专属控件按模式显隐。
"""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                            QAction, QActionGroup, QFileDialog, QMessageBox,
                            QDialog, QComboBox, QLineEdit, QCheckBox, QLabel,
                            QFrame, QSplitter, QStackedWidget,
                            QTableWidget, QHeaderView, QPushButton, QApplication)

from ..core import registry, runner, persistence
from ..core.dataset import Dataset
from ..core.pipeline import Pipeline, BUILTIN_STEPS
from ..core.contracts import RunConfig
from ..core.demo import demo_names, load_demo
from .chain_page import ChainPage
from .dialogs import TargetDialog, PipelineDialog   # C7-7a：类体已抽 dialogs.py
from .inspector import MethodInspector
from .method_browser import MethodBrowser
from .param_panel import ParamPanel
from .worker import BatchWorker, SingleWorker
from . import theme


class StageRail(QFrame):
    """阶段导航：把数模工作流（看数据→选模型→看细节）做成带编号的竖排入口。

    与内容栈联动：点条目切页，程序切页也回写选中态（双向同步）。
    """

    def __init__(self, groups, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(4)
        self._btns: dict[int, QPushButton] = {}
        self._stack = None
        for gi, (gtitle, entries) in enumerate(groups):
            head = QLabel(f"—— {gtitle} ——")
            head.setObjectName("railHead")      # 主题定义见 theme.QSS
            lay.addWidget(head)
            for idx, title in entries:
                b = QPushButton(title)
                b.setObjectName("railBtn")
                b.setCheckable(True)
                b.setCursor(Qt.PointingHandCursor)
                b.clicked.connect(lambda _=False, k=idx: self._go(k))
                lay.addWidget(b)
                self._btns[idx] = b
            if gi < len(groups) - 1:
                lay.addSpacing(6)
        lay.addStretch(1)

    def bind(self, stack: QStackedWidget):
        self._stack = stack
        stack.currentChanged.connect(self._sync)
        self._sync(stack.currentIndex())

    def _go(self, idx):
        if self._stack is not None:
            self._stack.setCurrentIndex(idx)

    def _sync(self, idx):
        for k, b in self._btns.items():
            b.setChecked(k == idx)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        registry.load_builtin()
        self.setWindowTitle("ML Toolbox · 机器学习 + 优化调参")
        self.resize(1920, 1080)
        theme.apply_to(self)

        self.dataset: Dataset | None = None
        self.spec = None
        self.pipeline = Pipeline.default()
        self._fitted_pipeline = None
        self.records: list = []
        self._worker = None
        self._current_method = None
        self._cfg = RunConfig()

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        # Perspective 容器：0 = ML 工作区，1 = 优化工作区（兄弟框架，整区切换）
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ml_page())
        from .opt_page import OptWorkbench
        self.opt_page = OptWorkbench(self)
        self.stack.addWidget(self.opt_page)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("就绪 —— 打开数据或加载演示数据集开始")

    # ================================================== 顶栏
    def _build_header(self) -> QWidget:
        # 两行顶栏（字号 x2 后单行放不下）：行1 = 品牌 + 模式开关 + 共享数据工具；
        # 行2 = ML 专属控件（优化模式整行隐藏）
        bar = QWidget()
        bar.setObjectName("header")
        root_lay = QVBoxLayout(bar)
        root_lay.setContentsMargins(14, 8, 14, 8)
        root_lay.setSpacing(6)

        lay = QHBoxLayout()
        lay.setSpacing(10)
        root_lay.addLayout(lay)

        brand = QLabel("ML Toolbox")
        brand.setObjectName("brand")
        sub = QLabel("机器学习 · 优化调参")
        sub.setObjectName("brandSub")
        bw = QVBoxLayout()
        bw.setSpacing(0)
        bw.addWidget(brand)
        bw.addWidget(sub)
        lay.addLayout(bw)
        lay.addSpacing(8)

        # ---- 模式切换（兄弟框架的 UI 语义：顶层对等状态切换）
        self._mode_switch = theme.ModeSwitch(["机器学习", "优化调参"])
        self._mode_switch.mode_changed.connect(self._set_mode)
        lay.addWidget(self._mode_switch)
        lay.addStretch(1)

        # ---- 共享数据工具
        btn_open = QPushButton("打开数据…")
        btn_open.clicked.connect(self._open_file)
        lay.addWidget(btn_open)

        self._demo = QComboBox()
        self._demo.addItems(["演示数据…"] + demo_names())
        self._demo.setMinimumWidth(240)
        self._demo.activated.connect(self._load_demo)
        lay.addWidget(self._demo)

        btn_pipe = QPushButton("管道设置…")
        btn_pipe.clicked.connect(self._open_pipeline_dialog)
        lay.addWidget(btn_pipe)

        btn_hist = QPushButton("历史…")
        btn_hist.setToolTip("从 runs/ 回看历史运行（核心图基于存档工件重绘，无需重训）")
        btn_hist.clicked.connect(self._open_history)
        lay.addWidget(btn_hist)

        # ---- 行2：ML 专属控件（优化模式下隐藏，pitfalls O9：隐藏包装控件）
        ml_lay = QHBoxLayout()
        ml_lay.setSpacing(10)
        root_lay.addLayout(ml_lay)
        self._ml_only_widgets: list = []

        def _add(w):
            ml_lay.addWidget(w)
            self._ml_only_widgets.append(w)
            return w

        _add(QLabel("随机种子"))
        self._seed = QLineEdit("42")
        self._seed.setFixedWidth(92)
        _add(self._seed)

        self._cv_folds = QComboBox()
        self._cv_folds.addItems(["3", "5", "10"])
        self._cv_folds.setCurrentText("5")
        self._cv_folds.setFixedWidth(96)
        self._cv_folds.setEnabled(False)

        self._cv = QCheckBox("交叉验证")
        self._cv.setToolTip("开启后遍历额外跑 k 折 CV，对比视图显示 cv_*_mean±std（默认关，零开销）")

        def _toggle_cv(v):
            self._cv_folds.setEnabled(v)
            self._cfg.extras["cv_folds"] = int(self._cv_folds.currentText()) if v else 0
        self._cv.toggled.connect(_toggle_cv)
        self._cv_folds.currentTextChanged.connect(
            lambda t: self._cfg.extras.__setitem__("cv_folds", int(t)) if self._cv.isChecked() else None)
        _add(self._cv)
        _add(self._cv_folds)

        self._diag = QCheckBox("诊断")
        self._diag.setToolTip("零侵入：开启后方法把中间产物装进 diag 供检视（模式 7）")
        self._diag.toggled.connect(lambda v: setattr(self._cfg, "diag", v))
        _add(self._diag)

        btn_pred = QPushButton("对新数据预测…")
        btn_pred.setToolTip("用当前选中方法的已拟合模型预测新的 csv（同管道变换 -> 导出预测表）")
        btn_pred.clicked.connect(self._predict_on_new)
        _add(btn_pred)
        ml_lay.addStretch(1)

        # 兼容旧动作接口（测试/快捷键可 trigger 切模式）
        self._mode_ml = QAction("机器学习", self, checkable=True)
        self._mode_opt = QAction("优化调参", self, checkable=True)
        self._mode_ml.setChecked(True)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        grp.addAction(self._mode_ml)
        grp.addAction(self._mode_opt)
        self._mode_ml.triggered.connect(lambda: self._set_mode(0))
        self._mode_opt.triggered.connect(lambda: self._set_mode(1))
        return bar

    def _set_mode(self, idx: int):
        """Perspective 切换：主区域整体换 + ML 专属顶栏项显隐。"""
        self.stack.setCurrentIndex(idx)
        self._mode_switch.set_mode(idx)
        ml = idx == 0
        for w in self._ml_only_widgets:
            w.setVisible(ml)
        if not ml:
            self.opt_page.refresh_source_status()

    # ================================================== 主体
    def _build_ml_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        lay = QHBoxLayout()
        lay.setSpacing(8)

        self.browser = MethodBrowser()
        self.browser.selection_changed.connect(self._on_selection)
        self.browser.method_activated.connect(self._inspect_method)

        # ---- 中部：阶段导航 + 内容栈（数模工作流顺序 = 内部执行顺序）
        center = QHBoxLayout()
        center.setSpacing(8)
        self.tabs = QStackedWidget()
        self.tabs.setObjectName("centerStack")
        self.chain_page = ChainPage()
        self.compare_table = QTableWidget()
        from PyQt5.QtWidgets import QHeaderView
        self.compare_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.compare_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.compare_table.cellClicked.connect(self._on_compare_click)
        self.inspector = MethodInspector()
        from .gallery import CompareGallery
        self.gallery = CompareGallery()
        self.gallery.thumb_clicked.connect(self._show_from_gallery)
        from .data_page import DataPage
        self.data_page = DataPage()
        for w in (self.chain_page, self.data_page, self.compare_table,
                  self.gallery, self.inspector):
            self.tabs.addWidget(w)
        # 索引集中管理（pitfalls L14：勿再硬编码 setCurrentIndex 数字）
        self.TAB_CHAIN = 0
        self.TAB_DATA = 1
        self.TAB_COMPARE = 2
        self.TAB_GALLERY = 3
        self.TAB_INSPECT = 4

        self.rail = StageRail([
            ("看数据", [(self.TAB_CHAIN, "处理链"), (self.TAB_DATA, "数据检视")]),
            ("选模型", [(self.TAB_COMPARE, "对比视图"), (self.TAB_GALLERY, "核心图对比")]),
            ("看细节", [(self.TAB_INSPECT, "方法检视")]),
        ])
        self.rail.bind(self.tabs)
        center.addWidget(self.rail)
        center.addWidget(self.tabs, 1)
        center_w = QWidget()
        center_w.setLayout(center)

        self.params = ParamPanel()
        self.params.rerun_requested.connect(self._rerun_current)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.browser)
        splitter.addWidget(center_w)
        splitter.addWidget(self.params)
        splitter.setSizes([560, 860, 500])
        splitter.setCollapsible(1, False)
        lay.addWidget(splitter)
        outer.addLayout(lay, 1)

        # 底部：运行控制
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 2, 10, 6)
        self.btn_run = QPushButton("▶ 遍历运行所选方法")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run_batch)
        self._progress = QLabel("")
        self._progress.setObjectName("muted")
        bl.addWidget(self.btn_run)
        bl.addWidget(self._progress, 1)
        outer.addWidget(bottom)
        return page

    # ================================================== 数据
    def _on_selection(self, names):
        self.statusBar().showMessage(
            f"候选 {len(names)} 个方法 —— 点『遍历运行』或单击某方法试跑")

    def _open_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "打开数据", "", "数据文件 (*.csv *.xlsx *.parquet *.json)")
        if not p:
            return
        try:
            df = pd.read_csv(p) if p.lower().endswith((".csv", ".txt")) \
                else pd.read_excel(p) if p.endswith((".xlsx", ".xls")) \
                else pd.read_parquet(p) if p.endswith(".parquet") \
                else pd.read_json(p)
        except Exception as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return
        dlg = TargetDialog(self, df.columns)
        if dlg.exec_() != QDialog.Accepted:
            return
        target, time_col = dlg.result_()
        self._set_dataset(Dataset(df, name=os.path.basename(p),
                                  target=target or None,
                                  time_col=time_col or None, source=p))

    def _load_demo(self, idx):
        if idx <= 0:
            return
        name = self._demo.itemText(idx)
        try:
            self._set_dataset(load_demo(name))
        except Exception:
            QMessageBox.warning(self, "演示数据加载失败", traceback.format_exc(limit=4))

    def _set_dataset(self, ds: Dataset):
        self.dataset = ds
        self._rebuild_spec()
        self.statusBar().showMessage(
            f"数据 [{ds.name}] n={len(ds.frame)} "
            f"目标列={ds.target or '无（无监督）'}")

    def _rebuild_spec(self):
        if self.dataset is None:
            return
        try:
            # Train-only path: scaler/imputer/encoder/feature-select state is
            # fitted on the training partition only; test rows are transform.
            self._fitted_pipeline = self.pipeline.fit(
                self.dataset, diag=self._cfg.diag)
            self.spec = self._fitted_pipeline.spec
        except Exception:
            QMessageBox.warning(self, "管道执行失败", traceback.format_exc(limit=6))
            return
        self.chain_page.show_spec(self.spec, self.dataset.profile())
        self.data_page.show_spec(self.spec)
        self.opt_page.set_data_context(self.spec)
        self.tabs.setCurrentIndex(self.TAB_CHAIN)

    def _open_pipeline_dialog(self):
        dlg = PipelineDialog(self, self.pipeline)
        if dlg.exec_() == QDialog.Accepted:
            self.pipeline = dlg.result_pipeline()
            self._rebuild_spec()

    def _open_history(self):
        from .history import HistoryDialog, load_run_record
        dlg = HistoryDialog(self)
        picked = {}

        def _load(run_id, kind):
            picked["id"], picked["kind"] = run_id, kind
        dlg.load_requested.connect(_load)
        dlg.exec_()
        if "id" not in picked:
            return
        run_id, kind = picked["id"], picked["kind"]
        if kind == "优化":
            self._load_opt_record(run_id)
            return
        r = load_run_record(run_id)
        if r is None:
            return
        self._set_mode(0)
        self.records = [r]
        self._fill_compare_table()
        self._show_record(0)
        self.tabs.setCurrentIndex(self.TAB_INSPECT)
        self.statusBar().showMessage(
            f"已载入历史运行 {r.run_id}（{r.method}）—— 图基于存档工件重绘")

    def _load_opt_record(self, run_id: str):
        """优化运行回看：载入评估历史到调参工作区（M3a：opt 记录写盘即可看）。"""
        from ..opt import persistence as opt_persistence
        try:
            rec = opt_persistence.load_record(run_id)
        except Exception:
            self.statusBar().showMessage(f"优化记录载入失败：{run_id}")
            return
        self._set_mode(1)                      # 切到优化调参 perspective
        self.opt_page.show_record(rec)
        self.statusBar().showMessage(
            f"已载入优化运行 {run_id}（{rec.optimizer}）—— 评估历史回看")

    def _predict_on_new(self):
        """用当前检视方法的已拟合模型，对新数据文件预测并导出预测表。

        依赖本次会话的 result.est（历史载入的运行无 estimator，会提示重跑）。
        """
        rec = getattr(self, "_current_record", None)
        if rec is None or not getattr(rec.result, "est", None):
            QMessageBox.information(
                self, "需要已拟合模型",
                "请先在方法库点选一个方法完成一次运行（历史载入的运行不含模型，"
                "需重跑）。")
            return
        if self.dataset is None:
            QMessageBox.information(self, "缺少上下文", "先加载训练数据。")
            return
        p, _ = QFileDialog.getOpenFileName(
            self, "选择待预测数据", "", "数据文件 (*.csv *.xlsx *.parquet)")
        if not p:
            return
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".csv":
                new = pd.read_csv(p)
            elif ext in (".xlsx", ".xls"):
                new = pd.read_excel(p)
            else:
                new = pd.read_parquet(p)
            Xnew = (self._fitted_pipeline or self.pipeline).transform_new(
                new, target=self.dataset.target)
            m = registry.get(rec.method)
            pred = m.predict(Xnew, rec.result)
        except Exception as e:
            QMessageBox.warning(self, "预测失败", traceback.format_exc(limit=6))
            return
        out = pd.DataFrame({"row": np.arange(len(pred)), "pred": np.asarray(pred)})
        save, _ = QFileDialog.getSaveFileName(
            self, "导出预测表", os.path.splitext(os.path.basename(p))[0]
            + f"_pred_{rec.method}.csv", "CSV (*.csv)")
        if save:
            out.to_csv(save, index=False)
            self.statusBar().showMessage(
                f"已用 {rec.method} 预测 {len(pred)} 行 -> {os.path.basename(save)}")

    # ================================================== 运行
    def _seed_value(self) -> int:
        try:
            return int(self._seed.text() or 42)
        except ValueError:
            return 42

    def _run_batch(self):
        if self.spec is None:
            QMessageBox.information(self, "还没有数据",
                                    "先打开数据文件或加载演示数据集。")
            return
        names = self.browser.selected()
        if not names:
            QMessageBox.information(self, "未选择方法",
                                    "在左侧方法库勾选要遍历的方法。")
            return
        self._cfg.seed = self._seed_value()
        self.btn_run.setEnabled(False)
        self._worker = BatchWorker(names, self.spec, self._cfg, self)
        self._worker.progress.connect(
            lambda i, n, nm: self._progress.setText(f"{i+1}/{n} · {nm}"))
        self._worker.finished_ok.connect(self._on_batch_done)
        self._worker.failed.connect(self._on_worker_fail)
        self._worker.start()

    def _on_batch_done(self, recs):
        self.btn_run.setEnabled(True)
        self._progress.setText("")
        self.records = recs
        for r in recs:
            if r.result.ok:
                try:
                    persistence.save_record(r)
                except Exception:
                    pass
        self._fill_compare_table()
        self.gallery.show_records(recs, self._pages_for)
        self.tabs.setCurrentIndex(self.TAB_GALLERY)
        n_ok = sum(r.result.ok for r in recs)
        self.statusBar().showMessage(
            f"遍历完成：{n_ok}/{len(recs)} 成功，结果已存入 runs/")
        if recs:
            self._show_record(0)

    def _on_worker_fail(self, tb_text):
        self.btn_run.setEnabled(True)
        QMessageBox.warning(self, "运行异常", tb_text[:800])

    def _fill_compare_table(self):
        table = runner.compare_table(self.records)
        if table.empty:
            self.compare_table.clear()
            return
        from .table_view import render_dataframe
        render_dataframe(self.compare_table, table,
                         red_when=lambda c, t: c == "error" and bool(t))

    def _on_compare_click(self, row, _col):
        self._show_record(row)
        self.tabs.setCurrentIndex(self.TAB_INSPECT)

    def _show_from_gallery(self, idx):
        """点缩略图 -> 跳到该方法完整检视。"""
        self._show_record(idx)
        self.tabs.setCurrentIndex(self.TAB_INSPECT)

    # ================================================== 检视 / 调参
    def _inspect_method(self, name: str):
        """单击方法库：立刻单独运行并检视（遍历的最小单元）。"""
        self._current_method = name
        m = registry.get(name)
        self._cfg.seed = self._seed_value()
        self.params.set_method(m, self._cfg)   # 先回写右栏：无数据时也要显示所选方法
        if self.spec is None:
            self.statusBar().showMessage("先加载数据，再点方法试跑")
            return
        self._worker = SingleWorker(m, self.spec, self._cfg, self)
        self._worker.finished_ok.connect(lambda r: self._show_single(r))
        self._worker.failed.connect(self._on_worker_fail)
        self.btn_run.setEnabled(False)
        self._worker.start()

    def _show_single(self, rec):
        self.btn_run.setEnabled(True)
        self._current_record = rec
        idx = next((i for i, r in enumerate(self.records)
                    if r.run_id == rec.run_id), None)
        if idx is None:
            self.records.append(rec)
            idx = len(self.records) - 1
        self._fill_compare_table()
        self._show_record(idx)
        self.tabs.setCurrentIndex(self.TAB_INSPECT)

    def _pages_for(self, rec):
        """某条记录对应的检视页（方法声明页 + 兜底 + 共享页）。"""
        try:
            m = registry.get(rec.method)
        except KeyError:
            return []
        spec = self.spec
        if spec is None:      # 载入历史运行且当前无数据：用记录里的 target_kind
            from ..core.contracts import DataSpec
            spec = DataSpec(X=None, target_kind=rec.target_kind)
        pages = m.inspect_pages(self._cfg) or runner.auto_pages(m, spec)
        return list(pages) + MLMethod_summary_pages()

    def _show_record(self, idx: int):
        if not (0 <= idx < len(self.records)):
            return
        rec = self.records[idx]
        self._current_record = rec
        self._current_method = rec.method
        try:
            m = registry.get(rec.method)
        except KeyError:
            return
        self.inspector.show_record(rec, self._pages_for(rec))
        self.params.set_method(m, RunConfig(overrides=rec.config.get("overrides", {}),
                                            diag=rec.config.get("diag", False),
                                            seed=rec.config.get("seed", 42)))

    def _rerun_current(self):
        if not self._current_method or self.spec is None:
            return
        overrides, errs = self.params.collect()
        if errs:
            return
        self._cfg.overrides = overrides
        self._inspect_method(self._current_method)

    def closeEvent(self, event):
        """关窗前优雅收尾：中止在跑的 worker + 清理黑盒子进程。

        否则 QThread 仍在跑就被销毁 -> Qt5Core fail-fast(0xc0000409)，
        且黑盒子进程成孤儿继续占核（见 docs/pitfalls.md）。
        """
        import logging
        _log = logging.getLogger("ml_toolbox.app")
        try:
            if getattr(self, "_worker", None) and self._worker.isRunning():
                _log.info("关窗：中止 ML 侧 worker")
                if hasattr(self._worker, "request_stop"):
                    self._worker.request_stop()
                self._worker.wait(5000)
            op = getattr(self, "opt_page", None)
            if op is not None:
                op.shutdown()
        except Exception as e:
            _log.warning("关窗收尾异常：%s", e)
        _log.info("=== GUI 关闭 ===")
        event.accept()


def MLMethod_summary_pages():
    from ..core.contracts import MLMethod
    return MLMethod.summary_pages()
