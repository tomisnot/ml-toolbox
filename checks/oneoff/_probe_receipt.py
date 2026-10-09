# -*- coding: utf-8 -*-
"""临时探针：量 `describe_training` 默认回执的实际体积（调查 A6-摩擦1）。"""
import json
import sys

sys.path.insert(0, r'D:\code-nosync\Mechine learning')
from ml_mecha.engine import MLEngine

rid = sys.argv[1]
# `_query_training_detail` 内部不用 self ⇒ 传 None 也能跑（只为量体积）
got = MLEngine._query_training_detail(None, run_id=rid)
blob = json.dumps(got, ensure_ascii=False, default=str)
print('run_id =', rid)
print('JSON 字符数 =', len(blob), ' (≈ %.0f 千 token，按 3.5 字符/token 估)'
      % (len(blob) / 3.5 / 1000))
print('顶层键 =', sorted(got))
print('tags 数 =', len(got['tags']))
print('curve 条目数 =', len(got['curve']))
print('convergence 条目 =', [(c['tag'], c['kind']) for c in got['convergence']])
print('latest 键 =', sorted(got['latest']))
print('latest_steps 条目数 =', len(got.get('latest_steps') or {}))
for k in sorted(got, key=lambda k: -len(json.dumps(got[k], default=str))):
    print('   %-16s %8d 字符' % (k, len(json.dumps(got[k], ensure_ascii=False, default=str))))
print()
print('--- latest 片段 ---')
print(json.dumps(got['latest'], ensure_ascii=False, default=str)[:1200])
print('--- 一个 sign/agree 的 curve ---')
print(json.dumps(got['curve'].get('sign/agree'), ensure_ascii=False, default=str))
print('--- 一个 ph/std 的 curve ---')
print(json.dumps(got['curve'].get('ph/std'), ensure_ascii=False, default=str))
