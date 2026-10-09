# -*- coding: utf-8 -*-
"""临时探针：扫描 runs/ 下每个 run 的 tb 事件文件集（调查 A4：同一目录是否被追加）。"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(r'D:\code-nosync\Mechine learning\runs')

rows = []
for d in ROOT.iterdir():
    if not d.is_dir():
        continue
    tb = d / 'tb'
    if not tb.is_dir():
        continue
    files = sorted(p for p in tb.glob('events*'))
    rec = d / 'record.json'
    started = ''
    if rec.is_file():
        try:
            m = json.loads(rec.read_text(encoding='utf-8'))
            started = str(m.get('started_at') or m.get('config', {}).get('extras', {})
                          .get('run_id') or '')
        except Exception as e:
            started = 'ERR:%s' % e
    rows.append((d.name, len(files), started,
                 [(f.name.split('.')[-1], round(f.stat().st_size / 1e6, 2))
                  for f in files]))

rows.sort(key=lambda r: r[0])
print('目录总数(有 tb):', len(rows))
print('多事件文件的 run（>=2）:')
for name, n, started, fs in rows:
    if n >= 2:
        print('  %-28s n=%d  %s' % (name, n, fs))
print()
print('全部:')
for name, n, started, fs in rows:
    print('  %-28s n=%d %s' % (name, n, fs))
