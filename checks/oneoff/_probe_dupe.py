# -*- coding: utf-8 -*-
"""临时探针：看一个 tb/ 目录里有两个事件文件的 run（A4 的现场证据）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'D:\code-nosync\Mechine learning')

rid = sys.argv[1]
root = Path(r'D:\code-nosync\Mechine learning\runs') / rid
print('=== 目录', root)
for f in sorted((root / 'tb').glob('events*')):
    st = f.stat()
    print('   文件 %s  %.3f MB  mtime=%d' % (f.name, st.st_size / 1e6, st.st_mtime))
rec = root / 'record.json'
if rec.is_file():
    m = json.loads(rec.read_text(encoding='utf-8'))
    print('   record.method =', m.get('method'), '| config.seed =',
          (m.get('config') or {}).get('seed'), '| started_at =', m.get('started_at'))
    print('   config.extras.run_id =', ((m.get('config') or {}).get('extras') or {}).get('run_id'))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for f in sorted((root / 'tb').glob('events*')):
    acc = EventAccumulator(str(root / 'tb'))
    acc.Reload()
    print('   [共用目录读回] scalars =', acc.Tags().get('scalars'))
    break
# 逐文件单独读：文件名唯一 ⇒ 用临时目录软链太麻烦，改为读每个文件的 steps
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2
for f in sorted((root / 'tb').glob('events*')):
    vals = {}
    for ev in event_file_loader.EventFileLoader(str(f)).Load():
        if ev.HasField('summary'):
            for v in ev.summary.value:
                if v.HasField('simple_value'):
                    vals.setdefault(v.tag, []).append((ev.wall_time, v.simple_value))
    print('   --- 单文件 %s' % f.name)
    for tag in sorted(vals):
        pts = vals[tag]
        print('       %-16s n=%d  steps?=%s  首/末值=%.6g/%.6g  wall(首/末)=%.0f/%.0f'
              % (tag, len(pts), '无step字段', pts[0][1], pts[-1][1],
                 pts[0][0], pts[-1][0]))
