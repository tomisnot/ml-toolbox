# -*- coding: utf-8 -*-
"""优化调参工作区（同窗口新增，docs/优化定位.md §7）。

UI 层次映射内部执行流（重设计）：
  左配置区 = 四段卡片，顺序即 ask-and-tell 循环的装配顺序：
    ① 目标函数（类型堆叠页：合成/ML 调参/外部程序）
    ② 数据源（仅 AutoTuner 需要，跨框架依赖显式呈现）
    ③ 优化器（多选=顺序对比）+ 超参
    ④ 预算 / 种子 / 热启动 / 预检
  右观察区 = 检视页直播（上）+ 评估历史（下）
  底控制条 = ▶开始/⏸暂停/停止 + 进度 + 配置导入导出

多优化器 = 顺序跑（G4 遍历哲学），各自流式刷新，最后叠加收敛曲线对比。
"""
from __future__ import annotations

import logging
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
                            QSplitter, QComboBox, QPushButton, QLabel, QLineEdit,
                            QCheckBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QTabWidget, QFormLayout, QFrame,
                            QStackedWidget, QGroupBox, QMessageBox, QScrollArea)

from ..opt import registry as opt_registry
from ..opt.contracts import Budget
from ..opt.synth import synth_names, make_objective_from_synth
from ..opt.plots import default_pages
from .inspector import MethodInspector
from .param_form import ParamForm
from .worker import OptWorker
from . import theme

_LOG = logging.getLogger("ml_toolbox.opt.ui")


class _AutoStack(QStackedWidget):
    """高度跟随当前页：避免被最高的隐藏页撑出大片空白。"""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class OptWorkbench(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        opt_registry.load_builtin()
        theme.apply_to(self)
        self._worker: OptWorker | None = None
        self._records: dict[str, "object"] = {}     # optimizer -> OptRecord
        self._opt_instances: dict[str, "object"] = {}   # optimizer -> 实例（代理切片按记录取，修 M6 串台）
        self._queue: list[str] = []                 # 待跑优化器
        self._running = False
        self._build()

    # ================================================== 布局
    @staticmethod
    def _card(title: str) -> QGroupBox:
        g = QGroupBox(title)
        return g

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        lay = QHBoxLayout()
        lay.setSpacing(8)
        outer.addLayout(lay, 1)

        # ============ 左：配置区（四段卡片，顺序=执行装配顺序） ============
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)

        # ---- ① 目标函数（堆叠页：类型切换整页替换，无 hide/show 闪烁）
        g1 = self._card("① 目标函数")
        f1w = QVBoxLayout(g1)
        f1w.setSpacing(8)
        row = QHBoxLayout()
        row.addWidget(QLabel("类型"))
        self._obj_kind = QComboBox()
        self._add_items(self._obj_kind, [("合成函数", "synth"),
                                         ("ML 方法调参（AutoTuner）", "ml"),
                                         ("外部程序（黑盒进程）", "proc")])
        self._obj_kind.currentTextChanged.connect(self._on_obj_kind)
        row.addWidget(self._obj_kind, 1)
        f1w.addLayout(row)

        self._obj_stack = _AutoStack()
        # 页 0：合成函数
        p_synth = QWidget()
        fs = QFormLayout(p_synth)
        self._synth = QComboBox()
        self._synth.addItems(synth_names())
        self._synth.currentTextChanged.connect(self._on_synth_change)
        fs.addRow("函数", self._synth)
        # 页 1：ML 方法调参（接缝2）
        p_ml = QWidget()
        fm = QFormLayout(p_ml)
        from ..core import registry as ml_registry
        ml_registry.load_builtin()
        self._ml_method = QComboBox()
        self._ml_method.addItems([m.name for m in ml_registry.all_methods()
                                  if m.task == "supervised"])
        self._ml_cv = QComboBox()
        self._ml_cv.addItems(["3", "5", "10"])
        self._ml_cv.setCurrentText("3")
        self._ml_cv.setFixedWidth(128)
        fm.addRow("方法", self._ml_method)
        fm.addRow("CV 折数", self._ml_cv)
        # 页 2：外部程序（评估侧接入）——线性表单，批量字段成对组合行
        p_proc = QWidget()
        fg = QFormLayout(p_proc)
        fg.setVerticalSpacing(6)
        self._proc_mode = QComboBox()
        self._proc_mode.addItems(["stdout 单点（解析一个分数）",
                                  "JSON 批量（pts/res 文件，可并行）"])
        self._proc_mode.currentTextChanged.connect(lambda *_: self._on_proc_mode())
        self._proc_params = QLineEdit("a=0..5, b=0..5")
        self._proc_params.setPlaceholderText("寻优参数：名=下界..上界 或 名=甲|乙|丙")
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
        self._proc_cons.setPlaceholderText("S_ret>=0.99, kick>=0.5（可空）")
        self._proc_extra = QLineEdit("{}")
        self._proc_extra.setPlaceholderText('固定字段 JSON：{"pol":"椭圆偏"}')
        self._proc_stagger = QLineEdit("4")
        self._proc_stagger.setToolTip("相邻子进程启动间隔秒（防共享数据库首访竞争）")
        self._proc_timeout = QLineEdit("1800")
        self._proc_timeout.setToolTip("单点评估超时（秒）")
        self._lbl_score = QLabel("目标字段")
        self._lbl_cons = QLabel("约束")
        self._lbl_parse = QLabel("解析")
        self._lbl_stagger = QLabel("错峰/超时")

        def _combo_row(*ws):
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            for w in ws:
                h.addWidget(w, 1)
            return box

        r_score = _combo_row(self._proc_score, self._proc_max)
        r_cons = _combo_row(self._proc_cons, self._proc_extra)
        r_time = _combo_row(self._proc_stagger, self._proc_timeout)
        fg.addRow("模式", self._proc_mode)
        fg.addRow("参数", self._proc_params)
        fg.addRow("命令", self._proc_cmd)
        fg.addRow("目录", self._proc_cwd)
        fg.addRow(self._lbl_parse, self._proc_parse)
        fg.addRow(self._lbl_score, r_score)
        fg.addRow(self._lbl_cons, r_cons)
        fg.addRow(self._lbl_stagger, r_time)
        # stdout 单点模式独有 / JSON 批量模式独有（含标签与组合行）
        self._proc_single_w = [self._lbl_parse, self._proc_parse]
        self._proc_batch_w = [self._lbl_score, r_score,
                              self._lbl_cons, r_cons,
                              self._lbl_stagger, r_time]
        self._obj_stack.addWidget(p_synth)
        self._obj_stack.addWidget(p_ml)
        self._obj_stack.addWidget(p_proc)
        f1w.addWidget(self._obj_stack)
        self._obj_info = QLabel("")
        self._obj_info.setObjectName("muted")
        self._obj_info.setWordWrap(True)
        f1w.addWidget(self._obj_info)
        lv.addWidget(g1)

        # ---- ② 数据源（显式呈现跨框架依赖）
        g0 = self._card("② 数据源")
        f0 = QFormLayout(g0)
        f0.setFieldGrowthPolicy(2)   # AllFieldsGrow（PyQt5 枚举兼容）
        self._src_kind = QComboBox()
        self._add_items(self._src_kind, [("ML 内部数据集", "ml"),
                                         ("外部文件（csv/parquet）", "file")])
        self._src_kind.currentTextChanged.connect(lambda *_: self.refresh_source_status())
        f0.addRow("来源", self._src_kind)
        self._src_path = QLineEdit()
        self._src_path.setPlaceholderText("数据文件路径（外部程序写入的 csv/parquet）")
        self._src_path.editingFinished.connect(self.refresh_source_status)
        self._lbl_path = QLabel("路径")
        f0.addRow(self._lbl_path, self._src_path)
        self._src_target = QLineEdit("target")
        self._src_target.setFixedWidth(180)
        self._src_target.editingFinished.connect(self.refresh_source_status)
        self._lbl_target = QLabel("目标列")
        f0.addRow(self._lbl_target, self._src_target)
        self._src_status = QLabel("")
        self._src_status.setObjectName("srcStatus")
        self._src_status.setStyleSheet("color:#888;")   # 未配置（灰）
        self._src_status.setWordWrap(True)
        f0.addRow(self._src_status)
        lv.addWidget(g0)
        self._src_group = g0

        # ---- ③ 优化器（全宽双列：短名不截断；多选=顺序对比）
        g2 = self._card("③ 优化器")
        f2 = QVBoxLayout(g2)
        f2.setSpacing(4)
        hint = QLabel("多选 = 顺序对比")
        hint.setObjectName("muted")
        f2.addWidget(hint)
        self._opt_boxes: dict[str, QCheckBox] = {}
        names = opt_registry.names()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        for i, name in enumerate(names):
            o = opt_registry.get(name)
            cb = QCheckBox(o.display_name)
            cb.setToolTip(name)
            cb.setChecked(name in ("gp_bo", "random_search"))
            grid.addWidget(cb, i // 2, i % 2)
            self._opt_boxes[name] = cb
        f2.addLayout(grid)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("超参面板针对："))
        self._opt_sel = QComboBox()
        self._opt_sel.addItems(names)
        self._opt_sel.currentTextChanged.connect(self._on_opt_change)
        sel_row.addWidget(self._opt_sel, 1)
        f2.addLayout(sel_row)
        lv.addWidget(g2)

        # ---- ④ 预算 / 运行（次数+并行一行，停滞+种子+热启动预检各一行）
        g3 = self._card("④ 预算 / 运行")
        f3 = QFormLayout(g3)
        f3.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        q1 = QHBoxLayout()
        self._n_evals = QLineEdit("40")
        self._workers = QLineEdit("1")
        self._workers.setToolTip("并行评估进程数（外部程序 JSON 批量模式有效；"
                                 "1=串行。建议 = 物理核数 × 0.7~1.0）")
        q1.addWidget(QLabel("次数"))
        self._n_evals.setFixedWidth(112)
        q1.addWidget(self._n_evals)
        q1.addSpacing(10)
        q1.addWidget(QLabel("并行"))
        self._workers.setFixedWidth(92)
        q1.addWidget(self._workers)
        q1.addStretch(1)
        f3.addRow(q1)
        q2 = QHBoxLayout()
        self._stall = QLineEdit("0")
        self._stall.setToolTip("连续无改善次数，0=不启用")
        self._seed = QLineEdit("42")
        q2.addWidget(QLabel("停滞停止"))
        self._stall.setFixedWidth(112)
        q2.addWidget(self._stall)
        q2.addSpacing(10)
        q2.addWidget(QLabel("种子"))
        self._seed.setFixedWidth(92)
        q2.addWidget(self._seed)
        q2.addStretch(1)
        f3.addRow(q2)
        # 热启动：复用旧 run 的评估历史（黑盒未变时 = 免费观测，不重评）
        ws_row = QWidget()
        wl = QHBoxLayout(ws_row)
        wl.setContentsMargins(0, 0, 0, 0)
        self._warm = QLineEdit("")
        self._warm.setPlaceholderText("run_id / history.csv（可选）")
        self._warm.setToolTip(
            "把旧优化的评估历史直接喂给新优化器（objective 未变时有效）：\n"
            "ok 点计入预算、不重新评估；failed 点只入历史不喂模型。\n"
            "GP-BO / TPE 支持完整吸收；其余引擎忽略这些点（仍占预算）。")
        wl.addWidget(self._warm)
        wb = QPushButton("…")
        wb.setFixedWidth(48)
        wb.setToolTip("浏览选择旧 run 目录或 history.csv")
        wb.clicked.connect(self._pick_warm)
        wl.addWidget(wb)
        f3.addRow("热启动", ws_row)
        # N3 锚点预检：正式跑前评一个已知答案的点，接口漂移当场拦截
        self._pf_anchor = QLineEdit("")
        self._pf_anchor.setPlaceholderText("freq_mhz=7350, intensity=0.65")
        self._pf_anchor.setToolTip(
            "锚点预检：启动正式优化前，先用当前配置评估这个已知点。\n"
            "黑盒静默变更接口 / 解析失配 / cwd 错 -> 当场拦截，不烧预算。\n"
            "留空 = 不预检。")
        f3.addRow("预检锚点", self._pf_anchor)
        pf2 = QWidget()
        pf2l = QHBoxLayout(pf2)
        pf2l.setContentsMargins(0, 0, 0, 0)
        self._pf_expect = QLineEdit("")
        self._pf_expect.setFixedWidth(140)
        self._pf_expect.setPlaceholderText("期望")
        self._pf_expect.setToolTip("期望分数（原始方向，如 C_dual）；留空=仅连通性检查")
        self._pf_tol = QLineEdit("0.05")
        self._pf_tol.setFixedWidth(112)
        self._pf_tol.setToolTip("绝对容差")
        pf2l.addWidget(self._pf_expect)
        pf2l.addWidget(QLabel("±"))
        pf2l.addWidget(self._pf_tol)
        pf2l.addStretch(1)
        f3.addRow("预检期望", pf2)

        # ---- ⑤ 优化器超参（选中优化器的 param_schema）——委托 ParamForm（C8）
        g4 = self._card("⑤ 超参")
        self._pf_layout = QFormLayout(g4)
        self._pf_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self._pf = ParamForm(self._pf_layout)

        # ④⑤ 竖排（不再并排：⑤ 超参卡独立成行，左栏变窄、超高走滚动条）
        lv.addWidget(g3)
        lv.addWidget(g4)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(660)
        lay.addWidget(left_scroll)

        # ---- 右：观察区（C4：声明式装配，复用 ML 侧 MethodInspector）+ 历史表
        right = QSplitter(Qt.Vertical)
        self.inspector = MethodInspector()
        right.addWidget(self.inspector)

        self.hist_table = QTableWidget()
        self.hist_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.hist_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hist_table.setMaximumHeight(220)
        right.addWidget(self.hist_table)
        right.setSizes([520, 220])
        lay.addWidget(right, 1)

        # ---- 底部控制条：运行 + 进度 + 配置导入导出
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.btn_run = QPushButton("▶ 开始优化")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._start)
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_export = QPushButton("导出配置")
        self.btn_export.setToolTip("把左栏全部设置（目标/数据源/优化器/预算）存成 JSON，"
                                   "下次导入即可复现同一次运行")
        self.btn_export.clicked.connect(self._export_cfg)
        self.btn_import = QPushButton("导入配置")
        self.btn_import.setToolTip("从 JSON 配置文件恢复全部设置")
        self.btn_import.clicked.connect(self._import_cfg)
        self._prog = QLabel("选择目标函数与优化器，点开始")
        self._prog.setObjectName("muted")

        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.btn_pause)
        bottom.addWidget(self.btn_stop)
        bottom.addWidget(self._prog, 1)
        bottom.addWidget(self.btn_import)
        bottom.addWidget(self.btn_export)
        outer.addLayout(bottom)
        self._on_opt_change(self._opt_sel.currentText())   # 初始填充超参面板
        self._on_obj_kind()                                 # 初始堆叠页

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
        # 控件构建/回收委托 ParamForm（C8）；owner 门控（O8）保留在此层
        o = opt_registry.get(name)
        self._pf.build(o.param_schema, owner=name, show_default=True,
                       field_width=180)

    @property
    def _param_owner(self):
        return self._pf.owner

    @property
    def _param_widgets(self):
        return self._pf.widgets

    def _collect_opt_cfg(self, name):
        """面板超参只归属于面板当前显示的优化器；其余优化器用默认值。"""
        if name != self._pf.owner:
            return {}
        return self._pf.values()

    # ================================================== 配置导出 / 导入
    CFG_VERSION = 1

    @staticmethod
    def _add_items(cb, pairs):
        """填充下拉框并给每项挂稳定 key（userData）。

        M7：业务逻辑读 currentData()（key），不再用中文标签 startswith 判断
        ——改文案不改行为。配置导出仍存中文标签（人类可读 + 向后兼容）。
        """
        from PyQt5.QtCore import QVariant
        for text, key in pairs:
            cb.addItem(text, QVariant(key))

    def _combo_key(self, cb) -> str:
        d = cb.currentData()
        return d if isinstance(d, str) else ""

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
                       "workers": self._workers.text(),
                       "warm_start": self._warm.text().strip(),
                       "preflight_anchor": self._pf_anchor.text().strip(),
                       "preflight_expect": self._pf_expect.text().strip(),
                       "preflight_tol": self._pf_tol.text().strip()},
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
                     ("seed", self._seed), ("workers", self._workers),
                     ("warm_start", self._warm),
                     ("preflight_anchor", self._pf_anchor),
                     ("preflight_expect", self._pf_expect),
                     ("preflight_tol", self._pf_tol)):
            if k in b:
                w.setText(str(b[k]))
        opts = cfg.get("optimizers")
        if opts:
            for n, cb in self._opt_boxes.items():
                cb.setChecked(n in opts)
        for n, oc in (cfg.get("optimizer_cfg") or {}).items():
            if n not in self._opt_boxes or not isinstance(oc, dict):
                continue
            if n != self._pf.owner:
                self._opt_sel.setCurrentText(n)     # _on_opt_change 重建面板
            self._pf.set_values(oc)
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

    def _pick_warm(self):
        """热启动源选择：旧 run 目录 或 history.csv。"""
        import os
        from PyQt5.QtWidgets import QFileDialog
        p = QFileDialog.getExistingDirectory(self, "选择旧优化 run 目录")
        if not p:
            p, _ = QFileDialog.getOpenFileName(
                self, "或选择 history.csv", "", "CSV (*.csv)")
        if p:
            self._warm.setText(os.path.normpath(p))

    def _warm_source(self):
        """解析热启动输入 -> 路径（run_id / 目录 / csv），空 -> None。"""
        from ..opt import persistence as _op
        s = self._warm.text().strip()
        if not s:
            return None
        import os
        if os.path.exists(s):
            return s
        # 当作 run_id：拼到 opt 存档根目录
        cand = os.path.join(_op.RUNS_DIR, s)
        if os.path.isdir(cand):
            return cand
        raise ValueError(f"热启动源不存在：{s}（填 run_id / run 目录 / history.csv）")

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
        self._opt_instances.clear()
        self._warm_shown = False
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
        return self._combo_key(self._obj_kind) == "ml"

    def _current_source(self):
        """按数据源选择构造 DataSource。抛异常 = 未配置/不可读。"""
        from ..opt.sources import SpecSource, FileSource
        if self._combo_key(self._src_kind) == "ml":
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
        file_mode = self._combo_key(self._src_kind) == "file"
        for w in (self._src_path, self._lbl_path, self._src_target,
                  self._lbl_target):
            w.setVisible(file_mode)
        if not self._uses_source():
            return
        try:
            src = self._current_source()
            X, y = src.fetch()                   # 真读一次 = 连接测试
            self._src_status.setText("● " + src.describe() + f" · 特征 {X.shape[1]} 维")
            self._src_status.setStyleSheet("color:#2ca02c;")   # 已连接（绿）
        except Exception as e:
            self._src_status.setText("● " + str(e)[:160])
            self._src_status.setStyleSheet("color:#c0392b;")   # 读取失败（红）

    def _on_obj_kind(self, text=None):
        key = self._combo_key(self._obj_kind)     # M7：按稳定 key 分支，不看文案
        synth = key == "synth"
        ml = key == "ml"
        proc = key == "proc"
        # 堆叠页整页替换（_AutoStack 高度跟随当前页，不留空白）
        self._obj_stack.setCurrentIndex(0 if synth else 1 if ml else 2)
        self._obj_stack.updateGeometry()
        if proc:
            self._on_proc_mode()
        if synth:
            self._on_synth_change(self._synth.currentText())
        elif ml:
            self._obj_info.setText("目标 = 方法在数据源上的 CV 主指标；"
                                   "数据源在下方显式选择（内部数据集或外部文件）。")
        else:
            self._obj_info.setText("参数注入命令模板跑外部程序，从 stdout 解析分数。"
                                   "数据由程序自产，无需数据源。")
        self.refresh_source_status()

    def _on_proc_mode(self):
        """外部程序子模式：stdout 单点 vs JSON 批量（并行）。"""
        if self._obj_stack.currentIndex() != 2:
            return
        batch = self._proc_mode.currentIndex() == 1
        for w in self._proc_single_w:
            w.setVisible(not batch)
        for w in self._proc_batch_w:
            w.setVisible(batch)
        self._obj_stack.updateGeometry()
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
        """薄转发：文法解析已下沉 opt/paramgrammar（M8/C6）。

        保留此入口仅为兼容既有调用点；解析逻辑与单测都在 opt 侧。
        """
        from ..opt.paramgrammar import parse_space
        return parse_space(text)

    def _make_objective(self):
        key = self._combo_key(self._obj_kind)     # M7：按稳定 key 分支
        if key == "synth":
            return make_objective_from_synth(self._synth.currentText())
        if key == "ml":
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
        """薄转发：约束文法已下沉 opt/paramgrammar（M8/C6）。"""
        from ..opt.paramgrammar import parse_constraints
        return parse_constraints(text)

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
        self._opt_instances[name] = optimizer       # 按名存，代理页按记录取（M6）
        budget = Budget(n_evals=int(self._n_evals.text() or 40),
                        stall=int(self._stall.text() or 0))
        cfg = self._collect_opt_cfg(name)
        seed = int(self._seed.text() or 42)
        try:
            warm = self._warm_source()
        except ValueError as e:
            self._prog.setText(str(e))
            self._running = False
            self._reset_buttons()
            return
        _LOG.info("启动优化：optimizer=%s objective=%s budget=%s seed=%s warm=%s",
                  name, objective.name, budget.n_evals, seed, warm or "-")
        try:
            workers = int(self._workers.text() or 1)
        except ValueError:
            workers = 1
        try:
            pre = self._preflight_spec()
        except ValueError as e:
            self._prog.setText(f"预检配置错误：{e}")
            self._running = False
            self._reset_buttons()
            return
        self._worker = OptWorker(objective, optimizer, budget, cfg=cfg,
                                 seed=seed, workers=workers, warm_start=warm,
                                 preflight=pre, parent=self)
        self._worker.eval_done.connect(self._on_eval)
        self._worker.progress.connect(
            lambda msg: self._prog.setText(f"运行中：{name} · {msg}"))
        self._worker.preflight_done.connect(
            lambda ok, msg, n=name: self._on_preflight(ok, msg, n))
        self._worker.finished_ok.connect(self._on_one_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()
        self._prog.setText(f"运行中：{name}（剩余 {len(self._queue)} 个）")

    def _on_preflight(self, ok, msg, name):
        """预检结论：失败 = 中止整轮（不烧预算、复位运行态）。"""
        _LOG.info("锚点预检 %s：%s", "通过" if ok else "失败", msg)
        if ok:
            self._prog.setText(f"预检通过：{msg} ｜ 继续 {name}…")
            return
        self._prog.setText(f"预检失败，已中止：{msg}")
        self._queue = []
        self._running = False
        self._reset_buttons()

    def _preflight_spec(self):
        """预检三字段 -> (anchor_dict, expect, tol)；锚点空 -> None（不预检）。"""
        text = self._pf_anchor.text().strip()
        if not text:
            return None
        from ..opt.paramgrammar import parse_anchor
        anchor = parse_anchor(text)
        exp = self._pf_expect.text().strip()
        expect = float(exp) if exp else None
        try:
            tol = float(self._pf_tol.text().strip() or "0.05")
        except ValueError:
            tol = 0.05
        return (anchor, expect, tol)

    def _on_eval(self, record, i):
        self._records[record.optimizer] = record
        self._refresh_live(record)
        # 热启动首帧：把回放数 + 指纹校验结论顶到状态栏（脏历史当场可见）
        note = getattr(record, "warm_note", "")
        if note and not getattr(self, "_warm_shown", False):
            self._warm_shown = True
            self._prog.setText(f"{note} ｜ 继续评估中…")
        # 逐点检查点：中途崩溃不丢已完成评估（昂贵黑盒的保险）
        if i % 4 == 3 or i == 0:
            try:
                from ..opt import persistence
                persistence.save_record(record)
            except Exception:
                pass

    def show_record(self, record):
        """历史回看入口（M3a）：把一条已存档 OptRecord 灌进检视页。

        与实时跑不同：无优化器实例（_opt_instances 空），代理切片页会显示
        "仅 GP-BO 可用"占位——这是诚实的（存档不含 estimator/GP，同 ML 侧
        "图基于工件重绘"的口径）。收敛/平行坐标/探索图只吃 history，可完整回看。
        """
        self._records = {record.optimizer: record}
        self._refresh_live(record)
        note = getattr(record, "warm_note", "")
        self._prog.setText(
            f"回看优化运行 {record.run_id}（{record.optimizer} · "
            f"{len(record.history)} 次评估）"
            + (f" ｜ {note}" if note else ""))

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
        _LOG.error("优化线程异常：%s", tb)
        self._prog.setText("优化线程异常（详见 logs/ml_toolbox.log）")
        self._running = False
        self._reset_buttons()

    def _finish_all(self):
        _LOG.info("全部优化器完成：%s", list(self._records))
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
            self._worker.kill_inflight()      # 立即中断在飞黑盒进程
            _LOG.info("用户点击停止（worker=%s running=%s）",
                      self._worker.optimizer.name, self._worker.isRunning())
        self._queue.clear()
        self._prog.setText("停止中：等待当前批次收尾…")

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸ 暂停")

    def shutdown(self):
        """关窗前调用：中止 worker + 清理在飞黑盒子进程，避免孤儿进程/
        QThread 销毁竞态（Qt5Core fail-fast 0xc0000409）。"""
        if getattr(self, "_worker", None) and self._worker.isRunning():
            _LOG.info("窗口关闭：中止运行中的优化 worker")
            self._worker.request_stop()
            self._worker.set_pause(False)
            self._worker.kill_inflight()
            self._worker.wait(8000)           # 给批内子进程收尾时间

    # ================================================== 刷新
    def _refresh_live(self, record):
        """C4：页声明归优化器（inspect_pages），UI 只装配。

        串台防护不变：按 record.optimizer 取实例再声明（看 A 画 A 的 GP）。
        无实例（历史回看）走 default_pages——代理切片是实例自足页，
        回看时诚实缺席（同 ML 侧"存档不含 estimator"口径）。
        """
        o = self._opt_instances.get(record.optimizer)
        pages = o.inspect_pages(record) if o is not None \
            else default_pages(record)
        self.inspector.show_record(record, pages)
        self._refresh_history(record)

    def _refresh_history(self, record):
        h = record.history
        if h.empty:
            return
        from .table_view import render_dataframe
        render_dataframe(self.hist_table, h,
                         red_when=lambda c, t: c == "status" and t == "failed")

    def _refresh_compare(self):
        if not self._records:
            return
        from ..opt.runner import compare_records
        from .table_view import render_dataframe
        render_dataframe(self.hist_table,
                         compare_records(list(self._records.values())))
