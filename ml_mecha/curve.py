# -*- coding: utf-8 -*-
"""**曲线形状**的判读：把"一个点"变成"一条趋势"（`describe_training` 用它）。

## 为什么要有它（用户实测的误判，代价一轮 1500 epoch 的训练）

用户原话：*"我一直只读 `latest`（一个点），然后靠'跨 run 在固定 epoch 上比终值'推断
曲线已饱和，据此把 epoch 从 900 砍到 250。后来我拉了**同一 run 内的曲线**，发现
`fid/sub1` 从 0.367 一路升到 0.682 且**在 epoch 250 时仍在爬** —— **根本没收敛，
是严重欠训练**。"*

⇒ 问题不在"缺数据"（曲线一直在 TB 里），而在**看曲线的形状很贵**：要
`include_series` + 手工在多条曲线之间对齐。本模块把这件事变成三个便宜的回答：

1. `shape()` —— **形状**而不是端点：起点/末值/**末段斜率**/是否单调/平台占比/趋势判读；
2. `convergence_note()` —— "末段还在爬"就**直接说**（`先别下结论`）；
   而"loss 类在爬"是**发散**，与"指标在爬（欠训练）"**分开说**（别用一个词盖住两种病）；
3. `best_point()` —— 按**任意 tag** 挑最佳点（不只是 `val_loss`），
   因为 `best.pt` 是按 `val_loss` 选的，而"fid 早就平了、val_loss 还在抖"时，
   两者**不是同一个 epoch**（用户实测踩过这个口径不一致）。

## 纪律

- **不插值**：对齐时缺失处就是 `None`（插值会造出一段没算过的曲线）；
- **点数不够就说不够**（`trend="too_few"`），不硬判趋势；
- 相对容差判"平台"（`rel_tol`）：绝对值阈在 loss（几十）与 fid（0..1）之间不通用。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

#: 末段窗口默认多少点（用户实测的欠训练在末 15 点仍上升 ⇒ 这个窗口抓得到）。
TAIL = 15

#: 判"平台/单调"的相对容差（相对于该段的量级）。
REL_TOL = 1e-3

#: 末段净变化超过这个比例才算"在爬/在掉"（挡住数值噪声）。
TREND_FRAC = 0.01


def _floats(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for v in values or ():
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _median(values: list[float]) -> float:
    """中位数（自实现：不引 numpy —— 本模块只做纯 Python 的曲线算术）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def shape(steps: Sequence[Any], values: Sequence[Any], *,
          tail: int = TAIL, rel_tol: float = REL_TOL) -> dict[str, Any]:
    """一条曲线的**形状摘要**（端点 + 末段斜率 + 单调性 + 平台占比 + 趋势）。

    ⚠ "趋势"是**末段窗口内**的判断，不是对整条曲线的断言 ⇒ 回执里同时给
    `tail`（窗口）与 `n`（总点数），让读的人知道这个结论覆盖了哪一段
    （`dry_run` 的 verdict 就是在这里吃过亏：'前 N 步健康 ≠ N 步后健康'）。
    """
    ys = _floats(values)
    xs = [float(s) for s in _floats(steps)] or list(range(len(ys)))
    n = min(len(ys), len(xs))
    ys, xs = ys[:n], xs[:n]
    if n == 0:
        return {"n": 0, "first": None, "last": None, "delta": None,
                "tail": 0, "slope_tail": None, "tail_rise": None,
                "monotone_up_tail": None, "flat_frac": None, "trend": "no_data"}
    first, last = ys[0], ys[-1]
    out: dict[str, Any] = {"n": n, "first": first, "last": last,
                           "delta": last - first,
                           #: 全程相对变化（末值 vs 起点）——判"到底在学没有"用**它**，
                           #: 不能只看末段（见 A52：末段回升 + 全程暴跌 ≠ 发散）。
                           "whole_frac": ((last - first) / abs(first)) if first else None}
    if n < 3:
        out.update({"tail": n, "slope_tail": None, "tail_rise": None,
                    "monotone_up_tail": None, "flat_frac": None,
                    "trend": "too_few"})
        return out
    w = max(2, min(int(tail), n))
    ty, tx = ys[-w:], xs[-w:]
    scale = max(1e-12, abs(sum(ty) / len(ty)))
    tol = float(rel_tol) * scale
    diffs = [ty[i + 1] - ty[i] for i in range(len(ty) - 1)]
    flat = sum(1 for d in diffs if abs(d) <= tol)
    med = _median([abs(d) for d in diffs])
    span = (tx[-1] - tx[0]) or 1.0
    slope = (ty[-1] - ty[0]) / span
    rise = ty[-1] - ty[0]
    mono_up = all(d >= -tol for d in diffs)
    mono_down = all(d <= tol for d in diffs)
    #: 末段分三段取均值：**抗噪**的"在爬/在掉"判据。
    #: ⚠ 实测教训（2026-10-05）：真实曲线**不是单调的**——用户的 `fid/sub1` 末 15 点
    #:   是 0.667→0.689 的爬升，但中间有 ±0.005 的抖动 ⇒ 只认"严格单调"会判成
    #:   `mixed`，于是**恰好漏掉他要的那个警告**（判据在真实夹具上当场红）。
    #:   ⇒ 用"前三分位均值 vs 后三分位均值"作为抗噪判据，`monotone` 只作为旁证。
    third = max(1, len(ty) // 3)
    d3 = (sum(ty[-third:]) / third) - (sum(ty[:third]) / third)
    thr = TREND_FRAC * scale
    if rise > thr and (mono_up or d3 > thr):
        trend = "rising"
    elif -rise > thr and (mono_down or -d3 > thr):
        trend = "falling"
    elif abs(rise) <= thr:
        trend = "flat"
    else:
        trend = "mixed"
    out.update({"tail": w, "slope_tail": slope, "tail_rise": rise,
                "monotone_up_tail": bool(mono_up), "flat_frac": flat / len(diffs),
                "tail_third_delta": d3, "trend": trend,
                #: **末段窗口自己的**端点与相对变化 —— 文案里说"末段在爬"时，
                #: 括号里的数字必须是**这两个**（拿全程端点去配末段的结论，
                #: 会写出"在上升（136→29.86）"这种自相矛盾的话；用户实测抓到的）。
                "tail_first": ty[0], "tail_last": ty[-1],
                "tail_rise_frac": rise / scale,
                #: 噪声尺度（末段相邻差的中位绝对差）与"回升/噪声"倍数：
                #: **给读数的人做判断用**，不作为唯一闸门——实测它分不开
                #: "真欠训练"和"已收敛"（0.93 vs 0.35，阈值放哪儿都误伤）。
                "noise_mad": med, "noise_frac": med / scale,
                "rise_over_noise": (rise / (3.0 * med)) if med > 0 else None})
    return out


#: "越小越好"的**名字特征**（loss 家族）：在爬 ⇒ 发散。
_LOSS_LIKE = ("loss", "err", "nll", "mse", "mae", "rmse", "mape", "kl", "divergence")

#: "越大越好"的**名字特征**（质量指标家族）：在爬 ⇒ 还没到头（可能欠训练）。
_QUALITY_LIKE = ("fid", "famp", "acc", "accuracy", "auc", "f1", "precision",
                 "recall", "r2", "silhouette", "score")

#: 全程降幅超过这个比例 ⇒ 这个配置在**学**，末段小回升**不构成发散**（见 A52）。
WHOLE_DROP_FRAC = 0.2

#: 末段涨幅超过这个比例 ⇒ **无论全程如何**都算真发散（后期爆炸）。
BIG_TAIL_FRAC = 0.5


def _tokens(tag: str) -> list[str]:
    """把 tag 切成**词元**（`/`、`_`、`.`、`-` 都算分隔）。"""
    import re
    return [t for t in re.split(r"[^0-9A-Za-z]+", str(tag)) if t]


def _matches(token: str, pattern: str) -> bool:
    """词元级匹配（允许 `val_loss` 这类**前缀_特征**写法，但**不许子串乱撞**）。

    ## 为什么必须词元级（实测教训 2026-10-06）

    第一版用**子串**匹配 ⇒ `grad/dec.1.linea**r2**.weight` 命中了特征 `r2`
    ⇒ 一个**逐层梯度**标签被当成"越大越好的质量指标"，于是梯度范数在爬被判成
    "**可能欠训练**"（用户实测抓到的：33 个 `grad/*` 里恰好这一个被诊断）。
    ⇒ 词元级后 `linear2` 是 `linear2` 而不是 `r2`，逐层梯度一律落到"方向未知"。
    """
    t, p = token.lower(), pattern.lower()
    return t == p or t.startswith(p + "_") or t.endswith("_" + p)


def direction(tag: str) -> str:
    """这个 tag 是"越小越好"还是"越大越好"，还是**方向未知**？

    ## 为什么必须承认"方向未知"（实测教训）

    第一版我复用了核心的 `is_lower_better`（它是给**主指标排序**用的，只认
    rmse/mae/mape/silhouette_deficit）⇒ `val_loss` 被当成"越大越好"，
    于是**"loss 在爬"被报成了"欠训练"**。两者的处置完全相反（一个是配置要改，
    一个是再多跑几轮）——**用一个词盖住两种病**正是本仓反复在防的。
    ⇒ 这里自己判方向并且**词元级**匹配（见 `_matches`）；**判不出来就承认判不出来**：
    `grad_norm` / `ph/std` / `grad/<层名>` / `weights_norm/*` 这类"没有好坏方向"的量
    **不给结论**（趋势仍照给，只是不诊断）。

    ## 定论（用户要求"别两边都留一半"）

    **逐层梯度与范数类一律不给诊断** —— 理由：它们在爬既可能"还没学完"、
    也可能是"梯度爆炸的早期"，**方向本身没有好坏**；把没有方向的东西安上一个
    诊断，就是把噪声变成结论。趋势照给（`curve[tag]`），读数的人自己判断。
    """
    try:
        from ml_toolbox.core.contracts import is_lower_better
        if is_lower_better(str(tag)):
            return "lower"
    except Exception:                                          # noqa: BLE001
        pass
    toks = _tokens(tag)
    if any(_matches(t, p) for t in toks for p in _LOSS_LIKE):
        return "lower"
    if any(_matches(t, p) for t in toks for p in _QUALITY_LIKE):
        return "higher"
    return "unknown"


def _lower_is_better(tag: str) -> bool:
    """`best_point` 用的方向：**未知时按"越大越好"**（并把这个假设一起回出去）。"""
    return direction(tag) == "lower"


def convergence_note(tag: str, sh: Mapping[str, Any]) -> dict[str, Any]:
    """"末段还在动"要说清 —— 但**分开说三种情形**（都被用户实测打出来过）。

    - **指标在爬** ⇒ `underfit`（欠训练：**别按终值判断饱和**，延长 epoch 再下结论）；
    - **loss 类在爬、且全程也在涨** ⇒ `diverging`（发散：这个配置有问题）；
    - **loss 类末段小回升、但全程是大幅下降** ⇒ `noisy_tail`（**不构成发散**）；
    - 方向未知（`grad*` / `ph/std` …）或趋势不是"在爬" ⇒ **不给结论**。

    ## 两条实测教训（都在这里修）

    1. **末段回升 ≠ 发散**（用户 2026-10-06 的 bug 报告）：他的 `loss` 全程
       1.92e+04 → 29.86（**降 99.8%**），只是末 15 点有 +1.5% 的噪声回升
       （噪声尺度 8.8%！），却被报成 `diverging` ⇒ 差点让他以为配置在发散。
       ⇒ 加**全程闸门**：全程显著下降时，末段小回升最多是 `noisy_tail`；
       只有"末段涨过 `BIG_TAIL_FRAC`"才**无论历史**都算真发散（后期爆炸）。
    2. **文案必须与数字一致**：旧版说"末段在上升"却印**全程**的 `first→last`
       ⇒ 写出了"在上升（136→29.86）"这种自相矛盾的话（用户原话："这种自相矛盾的
       文案比没有文案更糟"）。⇒ 现在"末段"那句印的是**末段窗口自己的**端点，
       全程数字单独说、带"全程"两个字。
    """
    trend = str(sh.get("trend") or "")
    if trend not in ("rising",):
        return {"kind": "", "note": ""}
    tail = int(sh.get("tail") or 0)
    t_first, t_last = sh.get("tail_first"), sh.get("tail_last")
    whole_frac = sh.get("whole_frac")
    tail_frac = sh.get("tail_rise_frac")
    where = direction(str(tag))
    if where == "lower":
        dropped = (whole_frac is not None and whole_frac <= -WHOLE_DROP_FRAC)
        blowup = (tail_frac is not None and tail_frac >= BIG_TAIL_FRAC)
        if dropped and not blowup:
            return {"kind": "noisy_tail",
                    "note": (f"⚠ {tag} **本窗口（{int(sh.get('n') or 0)} 点）**从 "
                             f"{_fmt(sh.get('first'))} 降到 {_fmt(sh.get('last'))}"
                             f"（{_fmt_pct(whole_frac)}）——末 {tail} 个点的 "
                             f"{_fmt_pct(tail_frac)} 回升"
                             f"（噪声尺度约 {_fmt_pct(sh.get('noise_frac'))}）"
                             f"**不构成发散**，只是抖动；要看更细就用 "
                             f"`include_series` 拉序列，或看验证集")}
        return {"kind": "diverging",
                "note": (f"⚠ {tag} 在末 {tail} 个评估点仍在**上升**（"
                         f"{_fmt(t_first)}→{_fmt(t_last)}，{_fmt_pct(tail_frac)}）"
                         f"⇒ 这个配置**可能在发散**（{tag} 应当下降）；"
                         f"⚠ 本窗口（{int(sh.get('n') or 0)} 点）："
                         f"{_fmt(sh.get('first'))}→{_fmt(sh.get('last'))}"
                         f"（{_fmt_pct(whole_frac)}）——先别开长训练，"
                         f"查学习率/归一化")}
    if where == "higher":
        return {"kind": "underfit",
                "note": (f"⚠ {tag} 在末 {tail} 个评估点仍在上升（"
                         f"{_fmt(t_first)}→{_fmt(t_last)}，{_fmt_pct(tail_frac)}）"
                         f"⇒ **可能欠训练**：别按终值判断'已饱和'，延长 epoch 再下结论"
                         f"（本窗口 {int(sh.get('n') or 0)} 点："
                         f"{_fmt(sh.get('first'))}→{_fmt(sh.get('last'))}，"
                         f"{_fmt_pct(whole_frac)}）")}
    return {"kind": "", "note": ""}


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):+.1%}"
    except (TypeError, ValueError):
        return "?"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "?"


def best_point(steps: Sequence[Any], values: Sequence[Any], *,
               lower_better: bool | None = None,
               tag: str = "") -> dict[str, Any]:
    """按这个 tag 挑**最佳点**（R41：不要只按 `val_loss` 挑）。

    ⚠ 为什么需要：`best.pt` 是按 `val_loss` 选的，而用户实测"同一 run 里 `val_loss`
    一直在 155–170 抖动，而 `fid` 早就平了"⇒ **两者不是同一个 epoch**，
    用它做 `eval_custom` 就是口径不一致。
    """
    ys = _floats(values)
    xs = [float(s) for s in _floats(steps)] or list(range(len(ys)))
    n = min(len(ys), len(xs))
    if n == 0:
        return {"step": None, "value": None, "index": None, "n": 0}
    d = _lower_is_better(tag) if lower_better is None else bool(lower_better)
    pick = min if d else max
    idx = pick(range(n), key=lambda i: ys[i])
    return {"step": xs[idx], "value": ys[idx], "index": idx, "n": n,
            "lower_better": d}


def align_tags(series: Mapping[str, Mapping[str, Any]], tags: Sequence[str], *,
               limit: int = 200) -> dict[str, Any]:
    """**同一个 run 内、多个 tag** 的对齐（R40 的同族，2026-10-06 补）。

    ## 为什么它比 `align`（多 run 同 tag）更紧急

    用户的实测 bug：他在一份回执里同时读到
    ``loss=0.018``、``fid/step=0.9965``、``fid/sub0=0.7301`` 三个数并判断
    "数学上不能同时成立"。而**三个数的求值步长本来就不一样**：
    `loss`/`fid/step` 是**逐 epoch**，`fid/sub0` 是**每 3 个 epoch**（`metrics_every`）。
    于是"每个 tag 各自的最新值"很可能**不在同一个 step 上** ——
    把它们并排读，就会得到一个**看着像矛盾的组合**（每个数在它自己那一步都是真的）。

    ⇒ 框架的职责是**把值放到同一根轴上**并如实标出缺失（`None`），
    而不是把不同步的值并排放着让人自己踩。`aligned_at_step` 给出
    **所有 tag 都有的最后一个 step**（那才是"可以放一起比"的那一步）。
    """
    picked = [str(t) for t in (tags or []) if str(t or "").strip()]
    per_tag: dict[str, dict[float, float]] = {}
    for tag in picked:
        got = (series or {}).get(tag) or {}
        ys = _floats(got.get("values"))
        xs = [float(s) for s in _floats(got.get("steps"))] or list(range(len(ys)))
        n = min(len(ys), len(xs))
        per_tag[tag] = {xs[i]: ys[i] for i in range(n)}
    axis = sorted({x for points in per_tag.values() for x in points})
    if limit and len(axis) > int(limit):
        axis = axis[-int(limit):]
    common = [x for x in axis if all(x in points for points in per_tag.values())]
    last_common = common[-1] if common else None
    return {
        "steps": axis,
        "tags": {tag: [points.get(x) for x in axis]
                 for tag, points in per_tag.items()},
        "n_points": {tag: len(points) for tag, points in per_tag.items()},
        #: **可以放一起比的那一步**（所有 tag 都有值的最后一个 step）；
        #: `None` ⇒ 没有公共 step（各 tag 的求值步长完全错开）⇒ 别并排读。
        "aligned_at_step": last_common,
        "aligned": ({tag: per_tag[tag][last_common] for tag in picked}
                    if last_common is not None else {}),
        "note": ("同一根 step 轴；某 tag 在那一步没记就写 null（**不插值**）。"
                 "⚠ `aligned_at_step` 才是「这些数能放一起比」的那一步——"
                 "各 tag 的求值步长可以不同（如 fid/* 每 N 轮一次）"),
    }


def align(series: Mapping[str, Mapping[str, Any]], tags: Sequence[str], *,
          limit: int = 200) -> dict[str, Any]:
    """多 run 曲线**对齐**（R40）：同一个 tag、同一根 step 轴、缺失处 `None`。

    `series` 形如 ``{run_id: {tag: {"steps": [...], "values": [...]}}}``。
    轴 = 各 run 该 tag 的 step **并集**（升序，取末尾 `limit` 个）；
    ⚠ **绝不插值**：某个 run 在那个 step 上没有点就写 `null`
    （插值会造出一段没算过的曲线，而"看起来更整齐"正是它危险的地方）。
    """
    out: dict[str, Any] = {}
    for tag in tags:
        per_run: dict[str, dict[float, float]] = {}
        for run_id, payload in (series or {}).items():
            got = (payload or {}).get(tag) or {}
            ys = _floats(got.get("values"))
            xs = [float(s) for s in _floats(got.get("steps"))] or list(range(len(ys)))
            n = min(len(ys), len(xs))
            per_run[str(run_id)] = {xs[i]: ys[i] for i in range(n)}
        axis = sorted({x for points in per_run.values() for x in points})
        if limit and len(axis) > int(limit):
            axis = axis[-int(limit):]
        out[tag] = {
            "steps": axis,
            "runs": {rid: [points.get(x) for x in axis]
                     for rid, points in per_run.items()},
            #: 有几个 run 在这个 tag 上真有数据（0 ⇒ 名字写错了/那次没记这个 tag）
            "n_runs": sum(1 for points in per_run.values() if points),
            "n_points": {rid: len(points) for rid, points in per_run.items()},
        }
    return {"tags": out, "note": "缺失处是 null（**不插值**）：那个 run 在那个 step "
                                 "上没有点，不是 0"}


def default_tags(series: Mapping[str, Mapping[str, Any]], *,
                 exclude_grad: bool = True, cap: int = 6) -> list[str]:
    """`compare` 没给 `tags` 时的默认：**至少两个 run 都有**的 tag（跳过逐层梯度）。

    ⚠ 跳过 `grad/...`：那是几十条逐层曲线，默认带上会把回执灌满（本仓反复
    在守"大字段默认有界"那条）。
    """
    counts: dict[str, int] = {}
    for payload in (series or {}).values():
        for tag in (payload or {}):
            if exclude_grad and str(tag).startswith("grad/"):
                continue
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    common = [t for t, c in counts.items() if c >= 2]
    common.sort(key=lambda t: (-counts[t], t))
    return common[:cap]


__all__ = ["shape", "convergence_note", "best_point", "align", "align_tags",
           "default_tags", "direction", "TAIL", "REL_TOL", "TREND_FRAC",
           "WHOLE_DROP_FRAC", "BIG_TAIL_FRAC"]
