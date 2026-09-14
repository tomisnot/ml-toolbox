# -*- coding: utf-8 -*-
"""主窗口：两种对等工作模式（Perspective）——ML 方法试验台 / 优化调参。

布局（简单模式原则：主界面只留必要控件，高级参数进语义位置面板）：
  工具栏   [ML 方法试验台 | 优化调参] 模式切换 · 打开文件 / 演示数据 /
           管道设置 / 历史 / 种子（ML 专属：对新数据预测 / 诊断 / 交叉验证）
  ML 模式   左方法库 | 中（处理链|数据|对比|画廊|检视）| 右参数面板 + 遍历条
  优化模式   OptWorkbench 全权接管（目标含合成/ML 调参/外部程序，数据源显式）

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
                            QComboBox, QLineEdit, QCheckBox, QLabel,
                            QSplitter, QTabWidget, QStackedWidget,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QPushButton, QApplication)

from ..core import registry, runner, persistence
from ..core.dataset import Dataset
from ..core.pipeline import Pipeline, BUILTIN_STEPS
from ..core.contracts import RunConfig
from ..core.demo import demo_names, load_demo
from .chain_page import ChainPage
from .inspector import MethodInspector
from .method_browser import MethodBrowser
from .param_panel import ParamPanel
from .worker import BatchWorker, SingleWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        registry.load_builtin()
        self.setWindowTitle("ML Toolbox · 方法试验台 + 优化调参")
        self.resize(1500, 900)

        self.dataset: Dataset | None = None
        self.spec = None
        self.pipeline = Pipeline.default()
        self.records: list = []
        self._worker = None
        self._current_method = None
        self._cfg = RunConfig()

        self._build_toolbar()
        self._build_body()
        self.statusBar().showMessage("就绪 —— 打开数据或加载演示数据集开始")

    # ================================================== 工具栏
    def _build_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)

        # ---- 模式切换（兄弟框架的 UI 语义：顶层对等，非 tab 附属）
        self._mode_ml = QAction("ML 方法试验台", self, checkable=True)
        self._mode_opt = QAction("优化调参", self, checkable=True)
        self._mode_ml.setChecked(True)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        grp.addAction(self._mode_ml)
        grp.addAction(self._mode_opt)
        self._mode_ml.triggered.connect(lambda: self._set_mode(0))
        self._mode_opt.triggered.connect(lambda: self._set_mode(1))
        tb.addAction(self._mode_ml)
        tb.addAction(self._mode_opt)
        tb.addSeparator()

        act_open = QAction("打开数据…", self)
        act_open.triggered.connect(self._open_file)
        tb.addAction(act_open)

        self._demo = QComboBox()
        self._demo.addItems(["演示数据…"] + demo_names())
        self._demo.activated.connect(self._load_demo)
        tb.addWidget(self._demo)

        act_pipe = QAction("管道设置…", self)
        act_pipe.triggered.connect(self._open_pipeline_dialog)
        tb.addAction(act_pipe)

        act_hist = QAction("历史…", self)
        act_hist.setToolTip("从 runs/ 回看历史运行（核心图基于存档工件重绘，无需重训）")
        act_hist.triggered.connect(self._open_history)
        tb.addAction(act_hist)

        # ML 专属控件（优化模式下隐藏）。注意：QToolBar 布局时会重新 show
        # 内嵌 widget，直接 hide(widget) 会被覆盖——必须 hide addWidget
        # 返回的包装 QAction（pitfalls O9）。
        self._ml_only_actions: list = []

        act_pred = QAction("对新数据预测…", self)
        act_pred.setToolTip("用当前选中方法的已拟合模型预测新的 csv（同管道变换 -> 导出预测表）")
        act_pred.triggered.connect(self._predict_on_new)
        tb.addAction(act_pred)
        self._ml_only_actions.append(act_pred)

        self._diag = QCheckBox("诊断")
        self._diag.setToolTip("零侵入：开启后方法把中间产物装进 diag 供检视（模式 7）")
        self._diag.toggled.connect(lambda v: setattr(self._cfg, "diag", v))
        self._ml_only_actions.append(tb.addWidget(self._diag))

        self._cv = QCheckBox("交叉验证")
        self._cv.setToolTip("开启后遍历额外跑 k 折 CV，对比视图显示 cv_*_mean±std（默认关，零开销）")
        self._cv_folds = QComboBox()
        self._cv_folds.addItems(["3", "5", "10"])
        self._cv_folds.setCurrentText("5")
        self._cv_folds.setFixedWidth(48)
        self._cv_folds.setEnabled(False)

        def _toggle_cv(v):
            self._cv_folds.setEnabled(v)
            self._cfg.extras["cv_folds"] = int(self._cv_folds.currentText()) if v else 0
        self._cv.toggled.connect(_toggle_cv)
        self._cv_folds.currentTextChanged.connect(
            lambda t: self._cfg.extras.__setitem__("cv_folds", int(t)) if self._cv.isChecked() else None)
        self._ml_only_actions.append(tb.addWidget(self._cv))
        self._ml_only_actions.append(tb.addWidget(self._cv_folds))

        self._ml_only_actions.append(tb.addWidget(QLabel("  种子 ")))
        self._seed = QLineEdit("42")
        self._seed.setFixedWidth(46)
        self._ml_only_actions.append(tb.addWidget(self._seed))

    def _set_mode(self, idx: int):
        """Perspective 切换：主区域整体换 + ML 专属工具栏项显隐。"""
        self.stack.setCurrentIndex(idx)
        ml = idx == 0
        for a in self._ml_only_actions:
            a.setVisible(ml)
        if not ml:
            self.opt_page.refresh_source_status()

    # ================================================== 主体
    def _build_body(self):
        # Perspective 容器：0 = ML 工作区，1 = 优化工作区（兄弟框架，整区切换）
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_ml_page())
        from .opt_page import OptWorkbench
        self.opt_page = OptWorkbench(self)
        self.stack.addWidget(self.opt_page)
        self.setCentralWidget(self.stack)

    def _build_ml_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)
        lay = QHBoxLayout()

        self.browser = MethodBrowser()
        self.browser.selection_changed.connect(self._on_selection)
        self.browser.method_activated.connect(self._inspect_method)

        self.tabs = QTabWidget()
        self.chain_page = ChainPage()
        self.compare_table = QTableWidget()
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
        self.tabs.addTab(self.chain_page, "处理链")
        self.tabs.addTab(self.data_page, "数据检视")
        self.tabs.addTab(self.compare_table, "对比视图")
        self.tabs.addTab(self.gallery, "核心图对比")
        self.tabs.addTab(self.inspector, "方法检视")
        # 索引集中管理（pitfalls L14：勿再硬编码 setCurrentIndex 数字）
        self.TAB_CHAIN = self.tabs.indexOf(self.chain_page)
        self.TAB_DATA = self.tabs.indexOf(self.data_page)
        self.TAB_COMPARE = self.tabs.indexOf(self.compare_table)
        self.TAB_GALLERY = self.tabs.indexOf(self.gallery)
        self.TAB_INSPECT = self.tabs.indexOf(self.inspector)

        self.params = ParamPanel()
        self.params.rerun_requested.connect(self._rerun_current)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.browser)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.params)
        splitter.setSizes([230, 950, 320])
        splitter.setCollapsible(1, False)
        lay.addWidget(splitter)
        outer.addLayout(lay, 1)

        # 底部：运行控制
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 2, 10, 6)
        self.btn_run = QPushButton("▶ 遍历运行所选方法")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2d6cdf;color:white;font-weight:bold;"
            "padding:9px 18px;border-radius:5px;font-size:13px;}"
            "QPushButton:hover{background:#1e54b8;}"
            "QPushButton:disabled{background:#aab;}")
        self.btn_run.clicked.connect(self._run_batch)
        self._progress = QLabel("")
        self._progress.setStyleSheet("color:#555;")
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
            self.spec = self.pipeline.run(self.dataset, diag=self._cfg.diag)
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
            Xnew = self.pipeline.transform_new(new, target=self.dataset.target)
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
        tw = self.compare_table
        tw.clear()
        if table.empty:
            return
        tw.setRowCount(len(table))
        tw.setColumnCount(len(table.columns))
        tw.setHorizontalHeaderLabels([str(c) for c in table.columns])
        for i, (_, row) in enumerate(table.iterrows()):
            for j, col in enumerate(table.columns):
                v = row[col]
                if pd.isna(v):
                    txt = ""
                elif isinstance(v, float):
                    txt = f"{v:.4g}"
                else:
                    txt = str(v)
                it = QTableWidgetItem(txt)
                if col == "error" and txt:
                    it.setForeground(Qt.red)
                tw.setItem(i, j, it)
        tw.resizeColumnsToContents()

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
        if self.spec is None:
            self.statusBar().showMessage("先加载数据，再点方法试跑")
            return
        self._current_method = name
        m = registry.get(name)
        self._cfg.seed = self._seed_value()
        self.params.set_method(m, self._cfg)
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


# ================================================== 对话框
from PyQt5.QtWidgets import QDialog, QDialogButtonBox


class TargetDialog(QDialog):
    """选择目标列 / 时间列。"""

    def __init__(self, parent, columns):
        super().__init__(parent)
        self.setWindowTitle("列角色")
        lay = QVBoxLayout(self)
        form = QHBoxLayout()
        form.addWidget(QLabel("目标列（y）："))
        self._t = QComboBox()
        self._t.addItems(["（无 / 无监督）"] + list(columns))
        form.addWidget(self._t, 1)
        lay.addLayout(form)
        form2 = QHBoxLayout()
        form2.addWidget(QLabel("时间列："))
        self._c = QComboBox()
        self._c.addItems(["（无）"] + list(columns))
        form2.addWidget(self._c, 1)
        lay.addLayout(form2)
        from ..core.contracts import REGRESSION_CARDINALITY
        lay.addWidget(QLabel(f"提示：数值且类别数 >{REGRESSION_CARDINALITY} "
                             "视为回归，否则分类。"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_(self):
        return ("" if self._t.currentIndex() == 0 else self._t.currentText(),
                "" if self._c.currentIndex() == 0 else self._c.currentText())


class PipelineDialog(QDialog):
    """管道步骤编辑：启用/禁用、参数覆写、顺序（上移/下移）。"""

    def __init__(self, parent, pipeline: Pipeline):
        super().__init__(parent)
        self.setWindowTitle("预处理管道")
        self.resize(560, 460)
        self._pipe = Pipeline(steps=[type(s)(enabled=s.enabled, **s.params)
                                     for s in pipeline.steps],
                              seed=pipeline.seed, test_size=pipeline.test_size,
                              stratify=pipeline.stratify,
                              time_split=pipeline.time_split)
        lay = QVBoxLayout(self)
        self._list = QTableWidget(len(self._pipe.steps), 4)
        self._list.setHorizontalHeaderLabels(["启用", "步骤", "参数(JSON)", ""])
        self._list.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._refresh_list()
        lay.addWidget(self._list, 1)

        row = QHBoxLayout()
        for nm, key in (("缺失值", "missing"), ("编码", "encode"),
                        ("缩放", "scale"), ("特征筛选", "feature_select"),
                        ("划分", "split")):
            b = QPushButton(f"+ {nm}")
            b.clicked.connect(lambda _=False, k=key: self._add_step(k))
            row.addWidget(b)
        lay.addLayout(row)

        form = QHBoxLayout()
        form.addWidget(QLabel("test_size"))
        self._ts = QLineEdit(str(self._pipe.test_size))
        self._ts.setFixedWidth(60)
        form.addWidget(self._ts)
        self._time = QCheckBox("按时间顺序切分（时序）")
        self._time.setChecked(self._pipe.time_split)
        form.addWidget(self._time)
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _refresh_list(self):
        self._list.setRowCount(len(self._pipe.steps))
        for i, st in enumerate(self._pipe.steps):
            chk = QCheckBox()
            chk.setChecked(st.enabled)
            chk.stateChanged.connect(lambda _s, idx=i: self._toggle(idx))
            self._list.setCellWidget(i, 0, chk)
            self._list.setItem(i, 1, QTableWidgetItem(
                f"{st.title} ({st.key})"))
            import json
            self._list.setItem(i, 2, QTableWidgetItem(
                json.dumps(st.params, ensure_ascii=False)))
            mv = QPushButton("删")
            mv.clicked.connect(lambda _=False, idx=i: self._del_step(idx))
            self._list.setCellWidget(i, 3, mv)

    def _toggle(self, idx):
        w = self._list.cellWidget(idx, 0)
        self._pipe.steps[idx].enabled = w.isChecked()

    def _add_step(self, key):
        self._pipe.steps.append(BUILTIN_STEPS[key]())
        self._refresh_list()

    def _del_step(self, idx):
        del self._pipe.steps[idx]
        self._refresh_list()

    def _accept(self):
        import json
        for i, st in enumerate(self._pipe.steps):
            raw = self._list.item(i, 2).text().strip()
            if raw:
                try:
                    st.params = json.loads(raw)
                except Exception:
                    QMessageBox.warning(self, "参数 JSON 非法",
                                        f"步骤 {st.key}: {raw}")
                    return
        try:
            self._pipe.test_size = float(self._ts.text())
        except ValueError:
            pass
        self._pipe.time_split = self._time.isChecked()
        self.accept()

    def result_pipeline(self) -> Pipeline:
        return self._pipe
