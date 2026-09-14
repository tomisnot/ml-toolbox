# -*- coding: utf-8 -*-
"""接线验证：新黑盒平面角度接口（1 点，占 1 核，可与正式跑并行）。

判据（交接文档 §6）：ω=7320, I=1.0, (α,θ,ε)=(90,0,0) 应复现纯 π 锚点
C_m05≈0.4285, C_p15≈0.2178, S_ret=1.0000（±0.01）。
这同时验证：① BatchProcessObjective 平面字段直通；② 黑盒角度语义正确。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.core.contracts import ParamSpec
from ml_toolbox.opt.contracts import ParamSpace
from ml_toolbox.opt.process import BatchProcessObjective

ROOT = r"D:\学\BaiduSyncdisk\编程\code\python\Energy Level"
space = ParamSpace([ParamSpec("freq_mhz", "f", "number", 7350.0, min=7200, max=7500),
                    ParamSpec("intensity", "I", "number", 0.65, min=0.3, max=1.0),
                    ParamSpec("alpha_deg", "a", "number", 0.0, min=0.0, max=90.0),
                    ParamSpec("theta_deg", "t", "number", 0.0, min=0.0, max=180.0),
                    ParamSpec("eps_deg", "e", "number", 0.0, min=-45.0, max=45.0)])
obj = BatchProcessObjective(
    "python scripts\\freeze_blackbox.py --points {points_file} --out {out_file} --fast",
    space, cwd=ROOT, point_extra={"fwhm_ns": 40.0},
    score_field="C_dual", minimize=False,
    constraints=[{"field": "S_ret", "op": ">=", "value": 0.99}],
    stagger=0.0, timeout=1200.0, name="wiring_check")

# 平面字段直通检查（不发废字段 pol/ell）
pt = obj._to_point({"freq_mhz": 7320.0, "intensity": 1.0,
                    "alpha_deg": 90.0, "theta_deg": 0.0, "eps_deg": 0.0}, 0)
assert "pol" not in pt and "ell" not in pt and pt["alpha_deg"] == 90.0, pt
print("点构造 OK:", pt)

import time
t0 = time.time()
s, st = obj.evaluate_many([{"freq_mhz": 7320.0, "intensity": 1.0,
                            "alpha_deg": 90.0, "theta_deg": 0.0,
                            "eps_deg": 0.0}])[0]
print(f"status={st} score(min方向)={s:.4f} 用时={time.time()-t0:.0f}s")
if st != "ok":
    print("last_error:", obj.last_error)
    sys.exit(1)
# 锚点判据：C_dual(90,0,0)≈0.2178（=min(0.4285,0.2178)）
c_dual = -s
print(f"C_dual={c_dual:.4f}（期望 0.2178±0.01）")
assert abs(c_dual - 0.2178) < 0.01, "未复现纯 π 锚点——接线或黑盒有问题！"
print("✅ 平面角度接口接线正确（纯 π 锚点复现）")
