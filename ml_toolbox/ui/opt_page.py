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

import os

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

        g0 = QGroupBox("数据源")
        f0 = QFormLayout(g0)
        # 数据从哪来（显式呈现跨框架/跨进程依赖）：内部数据集 / 外部文件
        self._src_kind = QComboBox()
        self._src_kind.addItems(["ML 内部数据集", "外部文件（csv/parquet）"])
        self._src_kind.currentTextChanged.connect(lambda *_: self.refresh_source_status())
        f0.addRow("来源", self._src_kind)
        self._src_path = QLineEdit()
        self._src_path.setPlaceholderText("数据文件路径（外部程序写入的 csv/parquet）")
        self._src_path.editingFinished.connect(self.refresh_source_status)
        self._lbl_path = QLabel("路径")
        f0.addRow(self._lbl_path, self._src_path)
        self._src_target = QLineEdit("target")
        self._src_target.setFixedWidth(90)
        self._src_target.editingFinished.connect(self.refresh_source_status)
        self._lbl_target = QLabel("目标列")
        f0.addRow(self._lbl_target, self._src_target)
        self._src_status = QLabel("")
        self._src_status.setStyleSheet("color:#888; font-size:11px;")
        self._src_status.setWordWrap(True)
        f0.addRow(self._src_status)
        form.addWidget(g0)
        self._src_group = g0

        g1 = QGroupBox("目标函数")
        f1 = QFormLayout(g1)
        self._obj_kind = QComboBox()
        self._obj_kind.addItems(["合成函数", "ML 方法调参（AutoTuner）",
                                 "外部程序（黑盒进程）"])
        self._obj_kind.currentTextChanged.connect(self._on_obj_kind)
        f1.addRow("类型", self._obj_kind)
        self._synth = QComboBox()
        self._obj_info = QLabel("")
        self._obj_info.setStyleSheet("color:#666; font-size:11px;")
        self._obj_info.setWordWrap(True)
        self._synth.addItems(synth_names())
        self._synth.currentTextChanged.connect(self._on_synth_change)
        f1.addRow("函数", self._synth)
        # ML 方法调参行（接缝2）：方法 + CV 折数
        from ..core import registry as ml_registry
        ml_registry.load_builtin()
        self._ml_method = QComboBox()
        self._ml_method.addItems([m.name for m in ml_registry.all_methods()
                                  if m.task == "supervised"])
        self._ml_cv = QComboBox()
        self._ml_cv.addItems(["3", "5", "10"])
        self._ml_cv.setCurrentText("3")
        self._lbl_method = QLabel("方法")
        self._lbl_cv = QLabel("CV 折数")
        f1.addRow(self._lbl_method, self._ml_method)
        f1.addRow(self._lbl_cv, self._ml_cv)
        # 外部程序行（评估侧接入）：参数定义 + 命令模板 + 解析式
        self._proc_mode = QComboBox()
        self._proc_mode.addItems(["stdout 单点（解析一个分数）",
                                  "JSON 批量（pts/res 文件，可并行）"])
        self._proc_mode.currentTextChanged.connect(lambda *_: self._on_proc_mode())
        self._lbl_mode = QLabel("模式")
        self._proc_params = QLineEdit("a=0..5, b=0..5")
        self._proc_params.setPlaceholderText("寻优参数：名=下界..上界 或 名=甲|乙|丙；"
                                             "名:点内路径=... 支持嵌套字段")
        self._proc_cmd = QLineEdit("python sim.py --a {a} --b {b}")
        self._proc_cmd.setPlaceholderText("单点：{key} 占位注入；批量：{points_file}/{out_file}")
        self._proc_cwd = QLineEdit()
        self._proc_cwd.setPlaceholderText("黑盒工作目录（可空）")
        self._proc_parse = QLineEdit(r"score[:=]\s*(-?\d+(?:\.\d+)?)")
        self._proc_parse.setPlaceholderText("stdout 正则（含一个捕获组）")
        self._proc_score = QLineEdit("score")
        self._proc_score.setPlaceholderText("res.json 目标字段（支持 a.b 嵌套）")
        self._proc_max = QCheckBox("越大越好")
        self._proc_max.setChecked(True)
        self._proc_cons = QLineEdit()
        self._proc_cons.setPlaceholderText("约束：S_ret>=0.99, kick>=0.5（逗号分隔，可空）")
        self._proc_extra = QLineEdit("{}")
        self._proc_extra.setPlaceholderText('固定字段 JSON：{"pol":"椭圆偏","fwhm_ns":40}')
        self._proc_stagger = QLineEdit("4")
        self._proc_stagger.setFixedWidth(70)
        self._proc_stagger.setToolTip("相邻子进程启动间隔秒（防共享数据库首访竞争）")
        self._proc_timeout = QLineEdit("1800")
        self._proc_timeout.setFixedWidth(70)
        self._proc_timeout.setToolTip("单点评估超时（秒）")
        self._lbl_pparams = QLabel("参数")
        self._lbl_cmd = QLabel("命令")
        self._lbl_cwd = QLabel("目录")
        self._lbl_parse = QLabel("解析")
        self._lbl_score = QLabel("目标字段")
        self._lbl_cons = QLabel("约束")
        self._lbl_extra = QLabel("固定字段")
        self._lbl_stagger = QLabel("错峰(s)")
        self._lbl_timeout = QLabel("超时(s)")
        f1.addRow(self._lbl_mode, self._proc_mode)
        f1.addRow(self._lbl_pparams, self._proc_params)
        f1.addRow(self._lbl_cmd, self._proc_cmd)
        f1.addRow(self._lbl_cwd, self._proc_cwd)
        f1.addRow(self._lbl_parse, self._proc_parse)
        f1.addRow(self._lbl_score, self._proc_score)
        f1.addRow(self._lbl_cons, self._proc_cons)
        f1.addRow(self._lbl_extra, self._proc_extra)
        f1.addRow("", self._proc_max)
        f1.addRow(self._lbl_stagger, self._proc_stagger)
        f1.addRow(self._lbl_timeout, self._proc_timeout)
        f1.addRow(self._obj_info)
        self._proc_single_w = [self._proc_parse]
        self._proc_batch_w = [self._proc_score, self._lbl_score, self._proc_cons,
                              self._lbl_cons, self._proc_extra, self._lbl_extra,
                              self._proc_max, self._proc_stagger, self._lbl_stagger]
        for w in (self._ml_method, self._lbl_method, self._ml_cv, self._lbl_cv,
                  self._proc_mode, self._lbl_mode,
                  self._proc_params, self._lbl_pparams,
                  self._proc_cmd, self._lbl_cmd,
                  self._proc_cwd, self._lbl_cwd,
                  self._proc_parse, self._lbl_parse,
                  self._proc_timeout, self._lbl_timeout,
                  *self._proc_batch_w):
            w.setVisible(False)
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
        self._workers = QLineEdit("1")
        self._workers.setFixedWidth(70)
        self._workers.setToolTip("并行评估进程数（外部程序 JSON 批量模式有效；"
                                 "1=串行。建议 = 物理核数 × 0.7~1.0）")
        f3.addRow("并行评估数", self._workers)
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
        self.btn_export = QPushButton("📤 导出配置")
        self.btn_export.setToolTip("把左栏全部设置（目标/数据源/优化器/预算）存成 JSON，"
                                   "下次导入即可复现同一次运行")
        self.btn_export.clicked.connect(self._export_cfg)
        self.btn_import = QPushButton("📥 导入配置")
        self.btn_import.setToolTip("从 JSON 配置文件恢复全部设置")
        self.btn_import.clicked.connect(self._import_cfg)
        self._prog = QLabel("选择目标函数与优化器，点开始")
        self._prog.setStyleSheet("color:#555;")

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.btn_pause)
        bottom.addWidget(self.btn_stop)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_import)
        bottom.addWidget(self.btn_export)
        outer.addLayout(bottom)
        outer2 = QHBoxLayout()
        outer2.addWidget(self._prog, 1)
        outer.addLayout(outer2)
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

    # ================================================== 配置导出 / 导入
    CFG_VERSION = 1

    def _combo_set(self, cb, text):
        i = cb.findText(str(text))
        if i >= 0:
            cb.setCurrentIndex(i)

    def get_config(self) -> dict:
        """左栏全部设置 -> 可 JSON 序列化的 dict（导出用）。"""
        cfg = {
            "kind": "ml_toolbox.opt.workspace",
            "version": self.CFG_VERSION,
            "objective_kind": self._obj_kind.currentText(),
            "synth": self._synth.currentText(),
            "ml_method": self._ml_method.currentText(),
            "ml_cv": self._ml_cv.currentText(),
            "source_kind": self._src_kind.currentText(),
            "source_path": self._src_path.text(),
            "source_target": self._src_target.text(),
            "proc": {
                "mode": self._proc_mode.currentIndex(),
                "params": self._proc_params.text(),
                "cmd": self._proc_cmd.text(),
                "cwd": self._proc_cwd.text(),
                "parse": self._proc_parse.text(),
                "score_field": self._proc_score.text(),
                "maximize": self._proc_max.isChecked(),
                "constraints": self._proc_cons.text(),
                "point_extra": self._proc_extra.text(),
                "stagger": self._proc_stagger.text(),
                "timeout": self._proc_timeout.text(),
            },
            "budget": {"n_evals": self._n_evals.text(),
                       "stall": self._stall.text(),
                       "seed": self._seed.text(),
                       "workers": self._workers.text()},
            "optimizers": self._selected_optimizers(),
            # 面板超参只归属当前选中的优化器（与 _launch_next 的运行语义一致）
            "opt_cfg_owner": getattr(self, "_param_owner", ""),
            "optimizer_cfg": {getattr(self, "_param_owner", ""):
                              self._collect_opt_cfg(self._param_owner)
                              if getattr(self, "_param_owner", "") else {}},
        }
        return cfg

    def set_config(self, cfg: dict):
        """get_config 的逆操作；未知键忽略，缺失键保持现值。"""
        if not isinstance(cfg, dict):
            raise ValueError("配置须是 JSON 对象")
        self._combo_set(self._obj_kind, cfg.get("objective_kind", "合成函数"))
        self._combo_set(self._synth, cfg.get("synth", ""))
        self._combo_set(self._ml_method, cfg.get("ml_method", ""))
        self._combo_set(self._ml_cv, cfg.get("ml_cv", ""))
        self._combo_set(self._src_kind, cfg.get("source_kind", ""))
        if "source_path" in cfg:
            self._src_path.setText(str(cfg["source_path"]))
        if "source_target" in cfg:
            self._src_target.setText(str(cfg["source_target"]))
        for k, w in (("mode", self._proc_mode), ("params", self._proc_params),
                     ("cmd", self._proc_cmd), ("cwd", self._proc_cwd),
                     ("parse", self._proc_parse),
                     ("score_field", self._proc_score),
                     ("constraints", self._proc_cons),
                     ("point_extra", self._proc_extra),
                     ("stagger", self._proc_stagger),
                     ("timeout", self._proc_timeout)):
            v = cfg.get("proc", {}).get(k)
            if v is None:
                continue
            if isinstance(w, QComboBox):
                w.setCurrentIndex(int(v))
            else:
                w.setText(str(v))
        if "maximize" in cfg.get("proc", {}):
            self._proc_max.setChecked(bool(cfg["proc"]["maximize"]))
        b = cfg.get("budget", {})
        for k, w in (("n_evals", self._n_evals), ("stall", self._stall),
                     ("seed", self._seed), ("workers", self._workers)):
            if k in b:
                w.setText(str(b[k]))
        opts = cfg.get("optimizers")
        if opts:
            for n, cb in self._opt_boxes.items():
                cb.setChecked(n in opts)
        for n, oc in (cfg.get("optimizer_cfg") or {}).items():
            if n not in self._opt_boxes or not isinstance(oc, dict):
                continue
            if n != self._param_owner:
                self._opt_sel.setCurrentText(n)     # _on_opt_change 重建面板
            for k, v in oc.items():
                pair = self._param_widgets.get(k)
                if pair is None:
                    continue
                p, w = pair
                if p.kind == "bool":
                    w.setChecked(bool(v))
                elif p.kind == "select":
                    self._combo_set(w, v)
                else:
                    w.setText(str(v))
        self._on_obj_kind(self._obj_kind.currentText())

    def _export_cfg(self):
        import json
        import os
        from PyQt5.QtWidgets import QFileDialog
        p, _ = QFileDialog.getSaveFileName(
            self, "导出优化配置", "opt_config.json", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.get_config(), f, ensure_ascii=False, indent=2)
            self._prog.setText(f"配置已导出：{os.path.basename(p)}")
        except OSError as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _import_cfg(self):
        import json
        from PyQt5.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, "导入优化配置", "", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            self.set_config(cfg)
            self._prog.setText(f"已导入配置：{p}")
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"{type(e).__name__}: {e}")

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

    def set_data_context(self, spec):
        """主窗口注入当前数据视图（"ML 内部数据集"数据源用）。"""
        self._spec = spec
        self.refresh_source_status()

    # ------------------------------------------------ 数据源（显式呈现依赖）
    def _uses_source(self) -> bool:
        """AutoTuner 需要数据；合成函数/外部程序不需要（进程自产数据）。"""
        return self._obj_kind.currentText().startswith("ML")

    def _current_source(self):
        """按数据源选择构造 DataSource。抛异常 = 未配置/不可读。"""
        from ..opt.sources import SpecSource, FileSource
        if self._src_kind.currentText().startswith("ML"):
            spec = getattr(self, "_spec", None)
            if spec is None or spec.y is None:
                raise RuntimeError("ML 内部数据集未加载（先在 ML 模式打开数据）")
            return SpecSource(spec)
        path = self._src_path.text().strip()
        if not path:
            raise RuntimeError("数据源=外部文件：请填写 csv/parquet 路径")
        return FileSource(path, self._src_target.text().strip() or "target")

    def refresh_source_status(self):
        """状态栏一行：绿=已连接 / 灰=未配置 / 红=读取失败。"""
        g0 = getattr(self, "_src_group", None)
        if g0 is None:
            return
        g0.setVisible(self._uses_source())
        file_mode = self._src_kind.currentText().startswith("外部")
        for w in (self._src_path, self._lbl_path, self._src_target,
                  self._lbl_target):
            w.setVisible(file_mode)
        if not self._uses_source():
            return
        try:
            src = self._current_source()
            X, y = src.fetch()                   # 真读一次 = 连接测试
            self._src_status.setText("● " + src.describe() + f" · 特征 {X.shape[1]} 维")
            self._src_status.setStyleSheet("color:#2ca02c; font-size:11px;")
        except Exception as e:
            self._src_status.setText("● " + str(e)[:160])
            self._src_status.setStyleSheet("color:#c0392b; font-size:11px;")

    def _on_obj_kind(self, text):
        synth = text.startswith("合成")
        ml = text.startswith("ML")
        proc = text.startswith("外部")
        self._synth.setVisible(synth)
        for w in (self._ml_method, self._lbl_method, self._ml_cv, self._lbl_cv):
            w.setVisible(ml)
        proc_widgets = [self._proc_mode, self._lbl_mode,
                        self._proc_params, self._lbl_pparams,
                        self._proc_cmd, self._lbl_cmd,
                        self._proc_cwd, self._lbl_cwd,
                        self._proc_timeout, self._lbl_timeout,
                        self._proc_parse, self._lbl_parse,
                        *self._proc_batch_w]
        for w in proc_widgets:
            w.setVisible(proc)
        if proc:
            self._on_proc_mode()
        if synth:
            self._on_synth_change(self._synth.currentText())
        elif ml:
            self._obj_info.setText("目标 = 方法在数据源上的 CV 主指标；"
                                   "数据源在上方显式选择（内部数据集或外部文件）。")
        else:
            self._obj_info.setText("参数注入命令模板跑外部程序，从 stdout 解析分数。"
                                   "数据由程序自产，无需数据源。")
        self.refresh_source_status()

    def _on_proc_mode(self):
        """外部程序子模式：stdout 单点 vs JSON 批量（并行）。"""
        if not self._proc_mode.isVisible():
            return
        batch = self._proc_mode.currentIndex() == 1
        self._proc_parse.setVisible(not batch)
        self._lbl_parse.setVisible(not batch)
        for w in self._proc_batch_w:
            w.setVisible(batch)
        if batch:
            self._workers.setText(self._workers.text() or "1")
            self._obj_info.setText("一批点写 pts.json → 黑盒进程 → 读 res.json。"
                                   "命令含 {points_file}/{out_file} 占位；"
                                   "并行评估数>1 时每点一个子进程（错峰启动）。")
        else:
            self._obj_info.setText("参数注入命令模板跑外部程序，从 stdout 解析分数。"
                                   "数据由程序自产，无需数据源。")

    @staticmethod
    def _proc_space(text: str):
        """参数定义 -> ParamSpace。

        语法（逗号分隔）：
          名=下界..上界          连续
          名=整数下界..上界#int  整数
          名=甲|乙|丙            离散 select
          点内路径=...           批量模式映射（如 ell.alpha_deg=-90..90，
                                 参数名 = 路径末段；亦支持 名:路径=...）
        """
        import re
        from ..core.contracts import ParamSpec
        from ..opt.contracts import ParamSpace
        specs, mapping = [], {}
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^([\w.]+)\s*(?::([\w.]+))?\s*=\s*(.+)$", tok)
            if not m:
                raise ValueError(f"参数定义无法解析：{tok!r}"
                                 "（期望 名=下界..上界 / 名=甲|乙 / 路径=...）")
            key, path, rhs = m.group(1), m.group(2), m.group(3).strip()
            if path is None and "." in key:     # 点路径即字段名，末段做参数名
                path, key = key, key.rsplit(".", 1)[-1]
            if path:
                mapping[key] = path
            mm = re.match(r"^(-?[\d.eE+-]+)\s*\.\.\s*(-?[\d.eE+-]+)(#int)?$", rhs)
            if mm:
                lo, hi = float(mm.group(1)), float(mm.group(2))
                if hi <= lo:
                    raise ValueError(f"参数 {key} 上界须大于下界")
                if mm.group(3):
                    specs.append(ParamSpec(key, key, "int", int((lo + hi) / 2),
                                           min=int(lo), max=int(hi)))
                else:
                    specs.append(ParamSpec(key, key, "number", (lo + hi) / 2,
                                           min=lo, max=hi))
            else:
                choices = [c.strip() for c in rhs.split("|") if c.strip()]
                if len(choices) < 2:
                    raise ValueError(f"参数 {key} 取值非法：{rhs!r}")
                specs.append(ParamSpec(key, key, "select", choices[0],
                                       choices=choices))
        if not specs:
            raise ValueError("至少定义一个寻优参数，如 a=0..5")
        return ParamSpace(specs), mapping

    def _make_objective(self):
        kind = self._obj_kind.currentText()
        if kind.startswith("合成"):
            return make_objective_from_synth(self._synth.currentText())
        if kind.startswith("ML"):
            from ..opt.bridges import AutoTunerObjective
            return AutoTunerObjective(
                self._ml_method.currentText(), self._current_source(),
                cv_folds=int(self._ml_cv.currentText()))
        space, mapping = self._proc_space(self._proc_params.text())
        cwd = self._proc_cwd.text().strip()
        if cwd and not os.path.isdir(cwd):
            raise ValueError(f"黑盒工作目录不存在：{cwd}"
                             "（导入的配置可能来自别的机器，请改「目录」）")
        if self._proc_mode.currentIndex() == 1:
            from ..opt.process import BatchProcessObjective
            import json as _json
            try:
                extra = _json.loads(self._proc_extra.text().strip() or "{}")
            except ValueError as e:
                raise ValueError(f"固定字段不是合法 JSON：{e}")
            if not isinstance(extra, dict):
                raise ValueError("固定字段须是 JSON 对象，如 {\"pol\": \"椭圆偏\"}")
            return BatchProcessObjective(
                self._proc_cmd.text().strip(), space,
                cwd=self._proc_cwd.text().strip(),
                point_map=mapping, point_extra=extra,
                score_field=self._proc_score.text().strip() or "score",
                minimize=not self._proc_max.isChecked(),
                constraints=self._parse_constraints(self._proc_cons.text()),
                stagger=float(self._proc_stagger.text() or 4),
                timeout=float(self._proc_timeout.text() or 1800),
                name="external_batch")
        from ..opt.process import ProcessObjective
        return ProcessObjective(self._proc_cmd.text().strip(), space,
                                parse=self._proc_parse.text().strip()
                                or r"score[:=]\s*(-?\d+(?:\.\d+)?)",
                                cwd=self._proc_cwd.text().strip(),
                                timeout=float(self._proc_timeout.text() or 120),
                                name="external_proc")

    @staticmethod
    def _parse_constraints(text: str):
        """"S_ret>=0.99, kick_m05>0.5" -> [{"field","op","value"}, ...]。"""
        import re
        out = []
        for tok in (text or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^([\w.]+)\s*(>=|<=|>|<|==)\s*(-?[\d.eE+-]+)$", tok)
            if not m:
                raise ValueError(f"约束无法解析：{tok!r}（期望 字段>=数值）")
            out.append({"field": m.group(1), "op": m.group(2),
                        "value": float(m.group(3))})
        return out

    def _launch_next(self):
        if not self._queue:
            self._finish_all()
            return
        name = self._queue.pop(0)
        try:
            objective = self._make_objective()
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
        try:
            workers = int(self._workers.text() or 1)
        except ValueError:
            workers = 1
        self._worker = OptWorker(objective, optimizer, budget, cfg=cfg,
                                 seed=seed, workers=workers, parent=self)
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
        # 留痕：runs/<run_id>/{opt_record.json, history.csv}（平坦性脚本/回看用）
        try:
            from ..opt import persistence
            d = persistence.save_record(record)
            self._prog.setText(f"已存留痕 → runs/{os.path.basename(d)}")
        except Exception as e:
            self._prog.setText(f"留痕失败：{e}")
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
