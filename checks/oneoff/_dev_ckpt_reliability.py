# -*- coding: utf-8 -*-
"""量化 `os.replace` 在 Windows 上的间歇性 `WinError 5`：`_atomic_torch_save` 的写盘可靠性。

P0 v2 探针在 222 次周期落盘里抓到 1 次 `PermissionError: [WinError 5]`（`os.replace`
换名那一步），说明"原子写"这一层不是 100% 成功。本探针把**失败率**量出来，
并给出"加一次重试能压到多少"的对照（只为判断值不值得改，不改源码）。

不改源码：`_atomic_torch_save` 与一个**加了重试**的本地副本在同一批数据上对比。
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(r"D:\code-nosync\Mechine learning")
TD = tempfile.mkdtemp(prefix="ck-reliab-")
os.environ["ML_TOOLBOX_RUNS"] = str(Path(TD) / "runs")
sys.path.insert(0, str(REPO))

import torch                                                              # noqa: E402

from ml_toolbox.methods.neural.custom import _atomic_torch_save           # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OBJ = {"state_dict": {f"layer{i}": torch.zeros(64, 64) for i in range(6)},
       "tag": "last", "saved_at_step": 1}


def _atomic_with_retry(obj, path, tries=3):
    """本地对照：同目录临时文件 + `os.replace`，失败**重试**（每次换一个临时名）。"""
    last = None
    for _ in range(tries):
        try:
            return _atomic_torch_save(obj, path)
        except OSError as exc:                       # noqa: PERM203
            last = exc
            time.sleep(0.002)
    raise last


def run(label, fn, n):
    d = Path(TD) / label
    d.mkdir(parents=True, exist_ok=True)
    p = d / "last.pt"
    t0 = time.time()
    fails, kinds = 0, {}
    for i in range(n):
        try:
            fn(OBJ, p)
        except BaseException as exc:                 # noqa: BLE001
            fails += 1
            kinds[type(exc).__name__ + ":" + str(getattr(exc, "winerror", ""))] = \
                kinds.get(type(exc).__name__ + ":" + str(getattr(exc, "winerror", "")), 0) + 1
    leftovers = [q.name for q in d.glob("*.tmp-*")]
    print(f"{label:>22}: {n} 次，失败 {fails}（{fails / n:.2%}），"
          f"耗时 {time.time() - t0:.1f}s，残留临时文件 {len(leftovers)}，失败分类 {kinds}")
    return fails


print(f"临时根 = {TD}")
print(f"对象大小 ≈ {sum(v.numel() for v in OBJ['state_dict'].values()) * 4 / 1024:.0f} KiB")
try:
    f1 = run("no_retry", _atomic_torch_save, N)
    f2 = run("with_retry_x3", _atomic_with_retry, N)
    print(f"\n结论：裸原子写失败率 {f1 / N:.2%}；加 3 次重试后 {f2 / N:.2%}")
finally:
    shutil.rmtree(TD, ignore_errors=True)
