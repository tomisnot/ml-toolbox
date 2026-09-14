# -*- coding: utf-8 -*-
"""外部程序接入的参数/约束文法（M8/C6：文本->领域对象的解析归 opt，不归 UI）。

分层纪律（架构审视 §9.3-5）：凡"文本 -> 领域对象"的解析一律落在 opt/ 或
core/，UI 只传文本、只显示错误。此前 `_proc_space` / `_parse_constraints`
长在 ui/opt_page.py，测试也直接断言 UI 私有方法——现下沉到本模块，UI 薄转发。

文法：
  parse_space("名=下界..上界, 名=甲|乙|丙, 点.内.路径=..., 名=1..9#int")
    -> (ParamSpace, {参数名: 点内路径})
  parse_constraints("S_ret>=0.99, kick>0.5")
    -> [{"field","op","value"}, ...]
"""
from __future__ import annotations

import re

from ..core.contracts import ParamSpec
from .contracts import ParamSpace

_SPACE_TOK = re.compile(r"^([\w.]+)\s*(?::([\w.]+))?\s*=\s*(.+)$")
_RANGE = re.compile(r"^(-?[\d.eE+-]+)\s*\.\.\s*(-?[\d.eE+-]+)(#int)?$")
_CONS_TOK = re.compile(r"^([\w.]+)\s*(>=|<=|>|<|==)\s*(-?[\d.eE+-]+)$")


def parse_space(text: str) -> tuple[ParamSpace, dict]:
    """参数定义文本 -> (ParamSpace, 点内路径映射)。

    语法（逗号分隔）：
      名=下界..上界          连续
      名=下界..上界#int      整数
      名=甲|乙|丙            离散 select
      点内路径=...           批量模式映射（如 ell.alpha_deg=-90..90，
                             参数名 = 路径末段；亦支持 名:路径=...）
    """
    specs, mapping = [], {}
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = _SPACE_TOK.match(tok)
        if not m:
            raise ValueError(f"参数定义无法解析：{tok!r}"
                             "（期望 名=下界..上界 / 名=甲|乙 / 路径=...）")
        key, path, rhs = m.group(1), m.group(2), m.group(3).strip()
        if path is None and "." in key:     # 点路径即字段名，末段做参数名
            path, key = key, key.rsplit(".", 1)[-1]
        if path:
            mapping[key] = path
        mm = _RANGE.match(rhs)
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


def parse_constraints(text: str) -> list:
    """"S_ret>=0.99, kick_m05>0.5" -> [{"field","op","value"}, ...]。"""
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = _CONS_TOK.match(tok)
        if not m:
            raise ValueError(f"约束无法解析：{tok!r}（期望 字段>=数值）")
        out.append({"field": m.group(1), "op": m.group(2),
                    "value": float(m.group(3))})
    return out
