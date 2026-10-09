# -*- coding: utf-8 -*-
"""临时探针：量"训练进行中"的可见性（轮询线程到底有没有拿到 CPU 片）。"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402
from ml_toolbox.core.persistence import RUNS_DIR          # noqa: E402

NET = open('tests/test_nqs_path.py', encoding='utf-8').read().split(
    "_NET_PERIODIC = '''")[1].split("'''")[0]

td = tempfile.mkdtemp(prefix='vis-')
ml = assemble_ml_mecha(root=td, seed=7)
ml.switch_ai()
try:
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(rng.randn(96, 4), columns=list('abcd'))
    frame['y'] = frame['a'] - 0.5 * frame['b']
    ds = ml.tools.execute('prepare_dataset', {
        'source': {'kind': 'frame', 'frame': frame}, 'name': 'p', 'target': 'y',
        'pipeline': {'steps': [{'key': 'scale'}]}})['value']['dataset_id']
    root = Path(RUNS_DIR)
    before = {p.name for p in root.iterdir() if p.is_dir()}
    out, seen, ticks = {}, {}, {'n': 0, 'alive': 0}
    stop = threading.Event()
    t0 = time.time()

    def _train():
        out['t_start'] = time.time() - t0
        out.update(ml.tools.execute('run_custom', {
            'device': 'cpu', 'code': NET, 'dataset_id': ds, 'name': 'p',
            'persist': True, 'checkpoint_every': 5,
            'overrides': {'epochs': 60, 'batch': 4, 'val_split': 0.25}}))
        out['t_end'] = time.time() - t0

    def _poll():
        while not stop.is_set():
            ticks['n'] += 1
            if th.is_alive():
                ticks['alive'] += 1
            try:
                for d in root.iterdir():
                    if d.name in before or d.name in seen:
                        continue
                    if (d / 'checkpoint' / 'last.pt').is_file():
                        seen[d.name] = (bool(th.is_alive()), round(time.time() - t0, 3))
            except OSError:
                pass
            time.sleep(0.002)

    th = threading.Thread(target=_train)
    th.start()
    poller = threading.Thread(target=_poll)
    poller.start()
    th.join(timeout=900)
    stop.set()
    poller.join(timeout=5)
    print('训练 t_start=%.2fs t_end=%.2fs（墙钟 %.2fs）'
          % (out.get('t_start', -1), out.get('t_end', -1), time.time() - t0))
    print('轮询 tick 数 =', ticks['n'], '其中"训练活着"的 =', ticks['alive'])
    print('seen =', seen)
    print('run_id =', (out.get('value') or {}).get('run_id'))
    print('checkpoint_every 回执 =', (out.get('value') or {}).get('checkpoint_every'))
finally:
    ml.close()
