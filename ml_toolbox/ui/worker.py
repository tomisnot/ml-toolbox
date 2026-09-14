# -*- coding: utf-8 -*-
"""后台运行线程：批量遍历不冻结界面。进度经信号回报主线程。

⚠ Windows 已知坑：LightGBM 与 Qt 的 OpenMP 冲突通过"import 顺序"解决
（见 methods/ensemble.py 顶层导入）——入口脚本必须在构造 QApplication
之前 import ml_toolbox.ui.main_window。
"""
from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from ..core import runner
from ..core.contracts import RunConfig, DataSpec


class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)          # done, total, current
    finished_ok = pyqtSignal(list)                # list[RunRecord]
    failed = pyqtSignal(str)

    def __init__(self, names: list[str], spec: DataSpec,
                 cfg: RunConfig, parent=None):
        super().__init__(parent)
        self.names, self.spec, self.cfg = names, spec, cfg

    def run(self):
        try:
            recs = runner.run_batch(
                self.names, self.spec, self.cfg,
                progress=lambda i, n, nm: self.progress.emit(i, n, nm))
            self.finished_ok.emit(recs)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SingleWorker(QThread):
    finished_ok = pyqtSignal(object)              # RunRecord
    failed = pyqtSignal(str)

    def __init__(self, method, spec: DataSpec, cfg: RunConfig, parent=None):
        super().__init__(parent)
        self.method, self.spec, self.cfg = method, spec, cfg

    def run(self):
        try:
            self.finished_ok.emit(runner.run_one(self.method, self.spec, self.cfg))
        except Exception:
            self.failed.emit(traceback.format_exc())


class OptWorker(QThread):
    """序贯优化后台线程（P13 直播）：每次评估 emit 一条，支持暂停/中止。

    与 BatchWorker 的关键差异：ML 是一次性 finished_ok；优化是过程流——
    eval_done 逐条携带截至当前的 OptRecord，UI 增量刷新收敛曲线。
    """
    eval_done = pyqtSignal(object, int)           # record, i
    progress = pyqtSignal(str)                    # 批内进度（黑盒点级完成）
    finished_ok = pyqtSignal(object)              # 最终 OptRecord
    failed = pyqtSignal(str)

    def __init__(self, objective, optimizer, budget, cfg=None, seed=42,
                 workers=1, warm_start=None, parent=None):
        super().__init__(parent)
        self.objective, self.optimizer, self.budget = objective, optimizer, budget
        self.cfg, self.seed = cfg or {}, seed
        self.workers = max(int(workers), 1)
        self.warm_start = warm_start
        self._stop = False
        self._pause = False

    def request_stop(self):
        self._stop = True

    def kill_inflight(self):
        """杀掉在飞黑盒子进程（停止/关窗）：让 evaluate_many 立即返回。"""
        try:
            k = getattr(self.objective, "kill_all", None)
            if k:
                k()
        except Exception:
            pass

    def set_pause(self, v: bool):
        self._pause = v

    def run(self):
        from ..opt.runner import optimize
        try:
            if hasattr(self.objective, "on_progress"):
                self.objective.on_progress = (
                    lambda done, total, p: self.progress.emit(
                        f"黑盒评估 {done}/{total} 完成"
                        + (f"（最新 ω={p.get('freq_mhz'):.1f} "
                           f"I={p.get('intensity'):.3f}）"
                           if "freq_mhz" in p else "")))
            rec = optimize(
                self.objective, self.optimizer, self.budget, cfg=self.cfg,
                seed=self.seed, workers=self.workers,
                warm_start=self.warm_start,
                on_eval=lambda r, i: self.eval_done.emit(r, i),
                should_stop=lambda: self._pause_wait())
            self.finished_ok.emit(rec)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _pause_wait(self) -> bool:
        """暂停=自旋等待（评估间隔短，可接受）；停止=立即 True。"""
        import time
        while self._pause and not self._stop:
            time.sleep(0.05)
        return self._stop
