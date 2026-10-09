# -*- coding: utf-8 -*-
"""临时探针：读 TB 事件，逐 tag 报点数（只读，调查用）。"""
import sys
sys.path.insert(0, r'D:\code-nosync\Mechine learning')
from ml_mecha.tb_series import read_run

for rid in sys.argv[1:]:
    p = read_run(rid)
    print('===', rid, p.get('available'), p.get('reason', ''))
    sc = p.get('scalars') or {}
    for t in sorted(sc):
        g = sc[t]
        print('   %-26s n=%6d latest_step=%s latest=%.6g'
              % (t, len(g['values']), g['latest']['step'], g['latest']['value']))
    print('   duplicate_steps:', p.get('duplicate_steps'))
