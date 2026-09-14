# -*- coding: utf-8 -*-
"""freeze 交付件：平坦性图 + 平台中心（交接文档 §5 验收项）。

搜索跑完（GUI 或后端）后，对最优候选在 (ω, I) 小窗上扫网格：
  - 画 C_dual 热图（分辨率默认 2.5 MHz × 5%，可调到 1 MHz × 5%）；
  - 提取 C_dual ≥ 阈值 的连通平台（含峰点），报告 ω/I 范围与平台中心；
  - PNG 存 runs/<run_id>/，控制台打印交付摘要。

用法：
  python benchmarks/freeze_flatness_report.py --run-id <runs下的目录名> \
      [--top 1] [--dw 15 --di 0.15 --nw 13 --ni 7 --thresh 0.9] \
      [--workers 8]

  --run-id 指向 runs/ 下的一次优化留痕（history.csv 提供候选点）；
  也可 --pts my.json 直接给候选点文件（[{"freq_mhz":...,"intensity":...}]）。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DEFAULT = os.path.join(ROOT, "cases", "freeze_autotune.json")


def build_objective(cfg_proc: dict, space_spec):
    """从 GUI 配置快照的 proc 段构造 BatchProcessObjective（后端复用）。"""
    from ml_toolbox.core.contracts import ParamSpec
    from ml_toolbox.opt.contracts import ParamSpace
    from ml_toolbox.opt.process import BatchProcessObjective
    sp = ParamSpace([ParamSpec(d["key"], d.get("label", d["key"]),
                               "number", (d["low"] + d["high"]) / 2,
                               min=d["low"], max=d["high"]) for d in space_spec])
    cons = []
    for tok in (cfg_proc.get("constraints") or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        import re
        m = re.match(r"^([\w.]+)\s*(>=|<=|>|<|==)\s*(-?[\d.eE+-]+)$", tok)
        if m:
            cons.append({"field": m.group(1), "op": m.group(2),
                         "value": float(m.group(3))})
    return BatchProcessObjective(
        cfg_proc["cmd"], sp, cwd=cfg_proc.get("cwd", ""),
        point_extra=json.loads(cfg_proc.get("point_extra") or "{}"),
        score_field=cfg_proc.get("score_field", "C_dual"),
        minimize=not cfg_proc.get("maximize", True),
        constraints=cons, on_infeasible="penalize",
        stagger=float(cfg_proc.get("stagger", 4) or 4),
        timeout=float(cfg_proc.get("timeout", 1800) or 1800),
        name="flatness")


def load_candidates(args):
    if args.pts:
        rows = json.load(open(args.pts, encoding="utf-8"))
        return pd.DataFrame(rows), None
    from ml_toolbox.opt import persistence
    rec = persistence.load_record(args.run_id)
    h = rec.history[rec.history["status"] == "ok"].copy()
    if h.empty:
        sys.exit("history 无 ok 点")
    # score 是最小化方向（maximize 时 = -C_dual），升序 = 从最好到最差
    h = h.sort_values("score")
    return h, args.run_id


def plateau(M, mask, i0, j0):
    """含 (i0,j0) 的 4-连通平台：返回 (行,列) 集合。"""
    from collections import deque
    seen = {(i0, j0)}
    q = deque([(i0, j0)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if (0 <= a < M.shape[0] and 0 <= b < M.shape[1]
                    and mask[a, b] and (a, b) not in seen):
                seen.add((a, b))
                q.append((a, b))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--pts", default="", help="候选点 json（替代 --run-id）")
    ap.add_argument("--config", default=CFG_DEFAULT,
                    help="GUI 配置快照（取 proc 段构造黑盒）")
    ap.add_argument("--top", type=int, default=1, help="前 K 个候选")
    ap.add_argument("--dw", type=float, default=15.0, help="ω 窗半宽 MHz")
    ap.add_argument("--di", type=float, default=0.15, help="I 相对半宽")
    ap.add_argument("--nw", type=int, default=13, help="ω 网格数")
    ap.add_argument("--ni", type=int, default=7, help="I 网格数")
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.run_id and not args.pts:
        sys.exit("需要 --run-id 或 --pts")
    cfg = json.load(open(args.config, encoding="utf-8"))
    proc = cfg["proc"]
    space_spec = [{"key": "freq_mhz", "low": 7200.0, "high": 7500.0},
                  {"key": "intensity", "low": 0.3, "high": 1.0},
                  {"key": "alpha_deg", "low": 0.0, "high": 90.0},
                  {"key": "theta_deg", "low": 0.0, "high": 180.0},
                  {"key": "eps_deg", "low": -45.0, "high": 45.0}]
    obj = build_objective(proc, space_spec)
    # 历史行里存在的搜索维度（2D 基线 run 只有前两列；5D run 五列全在）
    ANGLES = ("alpha_deg", "theta_deg", "eps_deg")

    hist, run_id = load_candidates(args)
    cands = []
    seen = set()
    for _, row in hist.iterrows():
        key = (round(float(row["freq_mhz"]), 1), round(float(row["intensity"]), 3))
        if key in seen:
            continue
        seen.add(key)
        cands.append(row)
        if len(cands) >= args.top:
            break

    out_dir = os.path.join(ROOT, "runs", run_id) if run_id else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    for rank, c in enumerate(cands):
        f0, i0 = float(c["freq_mhz"]), float(c["intensity"])
        # 偏振角取候选行自身值（2D run 缺列 -> 黑盒默认 (0,0,0)）
        fixed = {a: float(c[a]) for a in ANGLES if a in c.index
                 and pd.notna(c[a])}
        ws = np.clip(np.linspace(f0 - args.dw, f0 + args.dw, args.nw),
                     7200.0, 7500.0)
        ivs = np.clip(np.linspace(i0 * (1 - args.di), i0 * (1 + args.di),
                                  args.ni), 0.3, 1.0)
        plist = [{"freq_mhz": float(w), "intensity": float(I), **fixed}
                 for w in ws for I in ivs]
        ang_txt = (" ".join(f"{k.split('_')[0]}={v:g}" for k, v in fixed.items())
                   or "偏振默认(0,0,0)")
        print(f"候选#{rank}: ω={f0:.1f} I={i0:.3f} ({ang_txt}) -> 网格 "
              f"{args.nw}×{args.ni} = {len(plist)} 点（workers={args.workers}）…",
              flush=True)
        # 按 workers 分块并行评估；失败/不可行点记 NaN（热图上留白）
        raw = []
        for j in range(0, len(plist), max(args.workers, 1)):
            for s, st in obj.evaluate_many(plist[j:j + max(args.workers, 1)]):
                raw.append(s if st == "ok" else None)
        # 最小化方向 -> 还原 C_dual 原始方向（越大越好）
        scores = [(v if v is not None else np.nan) for v in raw]
        if proc.get("maximize", True):
            scores = [-v if np.isfinite(v) else v for v in scores]
        M = np.full((args.ni, args.nw), np.nan)
        for k, v in enumerate(scores):
            M[k // args.nw, k % args.nw] = v
        mask = M >= args.thresh
        if not np.isfinite(M).any():
            print("  ⚠ 网格全部失败/不可行，跳过该候选")
            continue
        pk = np.unravel_index(np.nanargmax(np.where(np.isnan(M), -np.inf, M)),
                              M.shape)
        if mask[pk]:
            cells = plateau(M, mask, *pk)
            ii = [a for a, _ in cells]
            jj = [b for _, b in cells]
            w_lo, w_hi = ws[min(jj)], ws[max(jj)]
            I_lo, I_hi = ivs[min(ii)], ivs[max(ii)]
            wc = float(np.mean(ws[jj]))
            Ic = float(np.mean(ivs[ii]))
        else:
            w_lo = w_hi = wc = f0
            I_lo = I_hi = Ic = i0
            cells = set()
        print(f"  峰值 C_dual={M[pk]:.4f} @ ω={ws[pk[1]]:.1f}, I={ivs[pk[0]]:.3f}")
        print(f"  平台（C≥{args.thresh}）: ω∈[{w_lo:.1f},{w_hi:.1f}] MHz, "
              f"I∈[{I_lo:.3f},{I_hi:.3f}]，{len(cells)} 格")
        print(f"  平台中心: ω={wc:.1f} MHz, I={Ic:.3f} mW/cm²  ← 交付这个点")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.pcolormesh(ws, ivs, np.where(np.isnan(M), M.min(), M),
                           shading="auto", cmap="viridis")
        cs = ax.contour(ws, ivs, M, levels=[args.thresh],
                        colors="white", linewidths=1.5)
        ax.clabel(cs, fmt=f"C≥{args.thresh:g}")
        ax.plot([f0], [i0], "r*", ms=14, label="搜索最优")
        ax.plot([wc], [Ic], "wo", mfc="none", ms=10, label="平台中心")
        ax.set_xlabel("freq (MHz)")
        ax.set_ylabel("intensity (mW/cm²)")
        ax.set_title(f"freeze 平坦性 · 候选#{rank}（nominal ω={f0:.1f}, I={i0:.2f}）")
        ax.legend(loc="lower right", fontsize=9)
        fig.colorbar(im, ax=ax, label="C_dual")
        fig.tight_layout()
        png = os.path.join(out_dir, f"flatness_{rank}.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  图 → {png}")
        df = pd.DataFrame({"freq_mhz": np.repeat(ws, args.ni),
                           "intensity": np.tile(ivs, args.nw),
                           "C_dual": [M[k // args.nw, k % args.nw]
                                      for k in range(len(plist))]})
        for a in ANGLES:
            if a in fixed:
                df[a] = fixed[a]
        df.to_csv(os.path.join(out_dir, f"flatness_{rank}.csv"), index=False)


if __name__ == "__main__":
    main()
