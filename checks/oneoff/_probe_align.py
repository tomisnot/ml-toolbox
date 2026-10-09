# -*- coding: utf-8 -*-
"""临时探针：同一 run 内把 epoch 级 tag 与 batch 级 tag 放到一根轴上会怎样（调查 A4）。"""
import json
import sys

sys.path.insert(0, r'D:\code-nosync\Mechine learning')
from ml_mecha.engine import MLEngine

rid = sys.argv[1]
tags = ['loss', 'val_loss', 'fid/step']
got = MLEngine._query_training_detail(None, run_id=rid, tags=tags, limit=300)
cmp_ = got['compare']
print('run_id =', rid)
print('aligned_at_step =', cmp_.get('aligned_at_step'))
print('n_points =', cmp_.get('n_points'))
print('aligned =', json.dumps(cmp_.get('aligned'), ensure_ascii=False))
ax = cmp_.get('steps') or []
print('轴长度 =', len(ax), ' 轴前 5 =', ax[:5], ' 轴后 5 =', ax[-5:])
# 取轴的中段，看两族 tag 在同一 x 上的值
mid = len(ax) // 2
lo, hi = max(0, mid - 3), mid + 3
print('轴中段 x =', ax[lo:hi])
for t in tags:
    print('   %-10s' % t, cmp_['tags'].get(t, [])[lo:hi])
print()
print('latest_steps =', json.dumps(got.get('latest_steps'), ensure_ascii=False))
print('该 tag 的曲线形状（fid/step）=',
      json.dumps(got['curve'].get('fid/step'), ensure_ascii=False))
print('convergence =', json.dumps(got['convergence'], ensure_ascii=False)[:600])
