# -*- coding: utf-8 -*-
"""优化调参工作区（同窗口新增，docs/优化定位.md §7）。

与 ML 工作区的镜像：左=目标函数+优化器库+预算+参数，右=检视页直播，
底=▶开始/⏸暂停/⏹停止。核心差异 = 过程流：eval_done 逐条刷新收敛曲线。

目标函数二选一（阶段2 先做合成函数；AutoTuner 在阶段4 接入）：
  ① 合成/解析函数（数模标定靶子：Ackley/Rosenbrock/…）
  ② ML 方法+数据集（预留，阶段4）

多优化器 = 顺序跑（G4 遍历哲学），各自流式刷新，最后叠加收敛曲线对比。
"""
from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
                            QComboBox, QPushButton, QLabel, QLineEdit,
                            QCheckBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QTabWidget, QFormLayout, QFrame,
                            QScrollArea, QGroupBox, QMessageBox)

from ..opt import registry as opt_registry
from ..opt.contracts import Budget
from ..opt.synth import synth_names, make_objective_from_synth
from .widgets import MplCanvas
from . import opt_plots
from .worker import OptWorker


class OptWorkbench(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        opt_registry.load_builtin()
        self._worker: OptWorker | None = None
        self._records: dict[str, "object"] = {}     # optimizer -> OptRecord
        self._queue: list[str] = []                 # 待跑优化器
        self._cur_opt = None                        # 当前优化器实例（代理切片用）
        self._running = False
        self._build()

    # ================================================== 布局
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)
        lay = QHBoxLayout()
        outer.addLayout(lay, 1)

        # ---- 左栏：目标 + 优化器 + 预算 + 参数
        left = QScrollArea()
        left.setWidgetResizable(True)
        host = QFrame()
        form = QVBoxLayout(host)
        form.setSpacing(10)

        g1 = QGroupBox("目标函数")
        f1 = QFormLayout(g1)
        self._obj_kind = QComboBox()
        self._obj_kind.addItems(["合成函数", "ML 方法调参（阶段4）"])
        f1.addRow("类型", self._obj_kind)
        self._synth = QComboBox()
        self._obj_info = QLabel("")
        self._obj_info.setStyleSheet("color:#666; font-size:11px;")
        self._obj_info.setWordWrap(True)
        self._synth.addItems(synth_names())
        self._synth.currentTextChanged.connect(self._on_synth_change)
        f1.addRow("函数", self._synth)
        f1.addRow(self._obj_info)
        self._on_synth_change(self._synth.currentText())
        form.addWidget(g1)

        g2 = QGroupBox("优化器（多选=顺序对比）")
        f2 = QVBoxLayout(g2)
        self._opt_boxes: dict[str, QCheckBox] = {}
        for name in opt_registry.names():
            o = opt_registry.get(name)
            cb = QCheckBox(f"{o.display_name}  ({name})")
            cb.setChecked(name in ("gp_bo", "random_search"))
            f2.addWidget(cb)
            self._opt_boxes[name] = cb
        self._opt_sel = QComboBox()
        self._opt_sel.addItems(opt_registry.names())
        self._opt_sel.currentTextChanged.connect(self._on_opt_change)
        f2.addWidget(QLabel("选中优化器（下方超参面板针对它）"))
        f2.addWidget(self._opt_sel)
        form.addWidget(g2)

        g3 = QGroupBox("预算 / 种子")
        f3 = QFormLayout(g3)
        self._n_evals = QLineEdit("40")
        self._n_evals.setFixedWidth(70)
        f3.addRow("评估次数", self._n_evals)
        self._stall = QLineEdit("0")
        self._stall.setFixedWidth(70)
        self._stall.setToolTip("连续无改善次数，0=不启用")
        f3.addRow("停滞停止", self._stall)
        self._seed = QLineEdit("42")
        self._seed.setFixedWidth(70)
        f3.addRow("种子", self._seed)
        form.addWidget(g3)

        # 参数子面板（选中优化器的 param_schema）
        g4 = QGroupBox("优化器超参")
        self._param_form = QFormLayout(g4)
        self._param_widgets: dict[str, tuple] = {}
        form.addWidget(g4)
        form.addStretch(1)
        left.setWidget(host)
        left.setFixedWidth(300)
        lay.addWidget(left)

        # ---- 右栏：检视页 + 历史表
        right = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.canvas_conv = MplCanvas(width=8, height=5)
        self.canvas_par = MplCanvas(width=8, height=5)
        self.canvas_sc = MplCanvas(width=8, height=5)
        self.canvas_pareto = MplCanvas(width=8, height=5)
        self.canvas_sur = MplCanvas(width=8, height=5)
        self.tabs.addTab(self._wrap(self.canvas_conv), "收敛曲线")
        self.tabs.addTab(self._wrap(self.canvas_par), "平行坐标")
        self.tabs.addTab(self._wrap(self.canvas_sc), "探索地图")
        self.tabs.addTab(self._wrap(self.canvas_pareto), "Pareto 前沿")
        self.tabs.addTab(self._wrap(self.canvas_sur), "代理切片")
        right.addWidget(self.tabs)

        self.hist_table = QTableWidget()
        self.hist_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.hist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hist_table.setMaximumHeight(220)
        right.addWidget(self.hist_table)
        right.setSizes([520, 220])
        lay.addWidget(right, 1)

        # ---- 底部控制条（覆盖到左栏下方）
        self.btn_run = QPushButton("▶ 开始优化")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2d6cdf;color:white;font-weight:bold;"
            "padding:8px 16px;border-radius:5px;}"
            "QPushButton:disabled{background:#aab;}")
        self.btn_run.clicked.connect(self._start)
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self._prog = QLabel("选择目标函数与优化器，点开始")
        self._prog.setStyleSheet("color:#555;")

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.btn_pause)
        bottom.addWidget(self.btn_stop)
        bottom.addWidget(self._prog, 1)
        outer.addLayout(bottom)
        self._on_opt_change(self._opt_sel.currentText())   # 初始填充超参面板

    def _wrap(self, w):
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(w)
        return host

    # ================================================== 联动
    def _on_synth_change(self, key):
        if not key:
            return
        try:
            sp, fn, fmin, xmin = __import__(
                "ml_toolbox.opt.synth", fromlist=["make_synth"]).make_synth(key)
            self._obj_info.setText(
                f"维度 {sp.dim} · 真最优 f*={fmin:.4g}\n"
                + "  ".join(f"{d['key']}∈[{d['low']:g},{d['high']:g}]"
                            for d in sp.describe()))
        except Exception as e:
            self._obj_info.setText(str(e))

    def _on_opt_change(self, name):
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._param_widgets.clear()
        self._param_owner = name              # 面板当前归属的优化器
        o = opt_registry.get(name)
        for p in o.param_schema:
            w = self._make_param(p)
            self._param_form.addRow(p.label or p.key, w)
            self._param_widgets[p.key] = (p, w)

    def _make_param(self, p):
        if p.kind == "bool":
            w = QCheckBox(); w.setChecked(bool(p.default)); return w
        if p.kind == "select":
            w = QComboBox(); w.addItems([str(c) for c in (p.choices or [])])
            return w
        w = QLineEdit(str(p.default)); w.setFixedWidth(90)
        w.setToolTip(p.hint or f"[{p.min},{p.max}]")
        return w

    def _collect_opt_cfg(self, name):
        """面板超参只归属于面板当前显示的优化器；其余优化器用默认值。"""
        if name != getattr(self, "_param_owner", None):
            return {}
        cfg = {}
        for k, (p, w) in self._param_widgets.items():
            if p.kind == "bool":
                cfg[k] = w.isChecked()
            elif p.kind == "select":
                cfg[k] = w.currentText()
            else:
                try:
                    cfg[k] = p.clean(w.text())
                except ValueError:
                    cfg[k] = p.default
        return cfg

    # ================================================== 运行控制
    def _selected_optimizers(self):
        return [n for n, cb in self._opt_boxes.items() if cb.isChecked()]

    def _start(self):
        if self._running:
            return
        opts = self._selected_optimizers()
        if not opts:
            QMessageBox.information(self, "未选优化器", "至少勾选一个优化器。")
            return
        self._records.clear()
        self._queue = list(opts)
        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self._launch_next()

    def _launch_next(self):
        if not self._queue:
            self._finish_all()
            return
        name = self._queue.pop(0)
        try:
            objective = make_objective_from_synth(self._synth.currentText())
        except Exception as e:
            self._prog.setText(f"目标构造失败：{e}")
            self._running = False
            self._reset_buttons()
            return
        optimizer = opt_registry.get(name)
        self._cur_opt = optimizer
        budget = Budget(n_evals=int(self._n_evals.text() or 40),
                        stall=int(self._stall.text() or 0))
        cfg = self._collect_opt_cfg(name)
        seed = int(self._seed.text() or 42)
        self._worker = OptWorker(objective, optimizer, budget, cfg=cfg,
                                 seed=seed, parent=self)
        self._worker.eval_done.connect(self._on_eval)
        self._worker.finished_ok.connect(self._on_one_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()
        self._prog.setText(f"运行中：{name}（剩余 {len(self._queue)} 个）")

    def _on_eval(self, record, i):
        self._records[record.optimizer] = record
        self._refresh_live(record)

    def _on_one_done(self, record):
        self._records[record.optimizer] = record
        self._refresh_live(record)
        self._launch_next()

    def _on_fail(self, tb):
        self._prog.setText("优化线程异常（见状态栏）")
        self._running = False
        self._reset_buttons()

    def _finish_all(self):
        self._running = False
        self._reset_buttons()
        self._refresh_compare()
        parts = []
        for k, v in self._records.items():
            if getattr(v, "multi", False):
                parts.append(f"{k}=前沿{len(v.pareto) if v.pareto is not None else 0}")
            elif v.best:
                parts.append(f"{k}={v.best['score']:.4g}")
            else:
                parts.append(f"{k}=失败")
        self._prog.setText(f"完成：{len(self._records)} 个优化器 · " + "  ".join(parts))

    def _toggle_pause(self):
        if self._worker:
            v = not self._worker._pause
            self._worker.set_pause(v)
            self.btn_pause.setText("▶ 继续" if v else "⏸ 暂停")

    def _stop(self):
        if self._worker:
            self._worker.request_stop()
            self._worker.set_pause(False)
        self._queue.clear()

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸ 暂停")

    # ================================================== 刷新
    def _refresh_live(self, record):
        if getattr(record, "multi", False):
            # 多目标：Pareto 页 + 平行坐标（按 f0 着色）；其余页占位
            self._draw(self.canvas_pareto, opt_plots.plot_pareto, record)
            self._draw(self.canvas_par, opt_plots.plot_parallel, record)
            for cv, msg in ((self.canvas_conv, "多目标无单一收敛曲线，看 Pareto 页"),
                            (self.canvas_sc, "多目标探索看 Pareto 前沿页"),
                            (self.canvas_sur, "多目标暂无代理切片")):
                cv.draw_result(lambda ax, _r, m=msg: ax.text(
                    0.5, 0.5, m, ha="center", va="center", color="#888"), None)
            self._refresh_history(record)
            return
        # 收敛（叠加已完成的多条）
        extra = [(k, r.history) for k, r in self._records.items()
                 if r is not record and len(r.history)]
        self._draw(self.canvas_conv, opt_plots.plot_convergence,
                   record, extra=extra)
        self._draw(self.canvas_par, opt_plots.plot_parallel, record)
        self._draw(self.canvas_sc, opt_plots.plot_scatter2d, record)
        self._refresh_surrogate(record)
        self._refresh_history(record)

    def _draw(self, canvas, fn, *args, **kw):
        try:
            canvas.draw_result(lambda ax, _r: fn(ax, *args, **kw), None)
        except Exception:
            pass

    def _refresh_surrogate(self, record):
        o = self._cur_opt
        if o is None or not hasattr(o, "surrogate_1d") or o._gpr is None:
            self.canvas_sur.draw_result(
                lambda ax, _r: ax.text(0.5, 0.5,
                    "代理切片仅 GP-BO 可用（且需 ≥1 次预热后）",
                    ha="center", va="center", color="#888"), None)
            return
        keys = [d["key"] for d in record.space_desc]
        if not keys:
            return
        best = record.best or {}
        other = {k: best.get(k) for k in keys if k != keys[0] and best.get(k) is not None}
        out = o.surrogate_1d(keys[0], other)
        if out is None:
            return
        xs, mu, sd = out
        ok = record.history[record.history["status"] == "ok"]
        obs_x = ok[keys[0]].to_numpy() if keys[0] in ok else np.array([])
        obs_y = ok["score"].to_numpy() if "score" in ok else np.array([])
        self._draw(self.canvas_sur, opt_plots.plot_surrogate_1d,
                   (xs, mu, sd, obs_x, obs_y, keys[0]))

    def _refresh_history(self, record):
        h = record.history
        if h.empty:
            return
        cols = list(h.columns)
        self.hist_table.clear()
        self.hist_table.setColumnCount(len(cols))
        self.hist_table.setHorizontalHeaderLabels(cols)
        self.hist_table.setRowCount(len(h))
        for i in range(len(h)):
            for j, c in enumerate(cols):
                v = h.iloc[i][c]
                txt = f"{v:.4g}" if isinstance(v, (int, float, np.floating)) \
                    else str(v)
                it = QTableWidgetItem(txt)
                if c == "status" and v == "failed":
                    it.setForeground(Qt.red)
                self.hist_table.setItem(i, j, it)

    def _refresh_compare(self):
        if not self._records:
            return
        from ..opt.runner import compare_records
        df = compare_records(list(self._records.values()))
        self.hist_table.clear()
        self.hist_table.setColumnCount(len(df.columns))
        self.hist_table.setHorizontalHeaderLabels(list(df.columns))
        self.hist_table.setRowCount(len(df))
        for i in range(len(df)):
            for j, c in enumerate(df.columns):
                v = df.iloc[i][c]
                txt = f"{v:.4g}" if isinstance(v, (int, float, np.floating)) \
                    else str(v)
                self.hist_table.setItem(i, j, QTableWidgetItem(txt))
