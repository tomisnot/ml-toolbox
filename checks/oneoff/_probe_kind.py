# -*- coding: utf-8 -*-
"""临时探针：为什么 24 行时 loss 拿到 Long target？打印 prep 的真相。"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402
import ml_toolbox.methods.neural.custom as C             # noqa: E402
from ml_toolbox.methods.neural import trainer as T        # noqa: E402

src = open(REPO / 'tests/test_nqs_path.py', encoding='utf-8').read()
NET = src.split("_NET_PERIODIC = '''")[1].split("'''")[0]

_orig = T.train


def spy(recorder, *, model, prep, torch, nn, **kw):
    print('   prep.kind =', prep.kind, '| yv.dtype =',
          getattr(prep.yv, 'dtype', None), '| n_out =', prep.n_out,
          '| Xs', prep.Xs.shape, '| tr', len(prep.tr_idx), 'va', len(prep.va_idx))
    return _orig(recorder, model=model, prep=prep, torch=torch, nn=nn, **kw)


C._trainer.train = spy
T.train = spy

for rows, seed, vs in ((24, 3, 0.0), (96, 0, 0.25), (24, 3, 0.25)):
    td = tempfile.mkdtemp(prefix='diag-')
    ml = assemble_ml_mecha(root=td, seed=7)
    ml.switch_ai()
    try:
        rng = np.random.RandomState(seed)
        frame = pd.DataFrame(rng.randn(rows, 4), columns=list('abcd'))
        frame['y'] = frame['a'] - 0.5 * frame['b'] + 0.1 * rng.randn(rows)
        print('=== rows=%d seed=%d val_split=%s | y.head=%s' % (
            rows, seed, vs, list(np.round(frame['y'].head(3).to_numpy(), 3))))
        ds = ml.tools.execute('prepare_dataset', {
            'source': {'kind': 'frame', 'frame': frame}, 'name': 'u', 'target': 'y',
            'pipeline': {'steps': [{'key': 'scale'}]}})['value']['dataset_id']
        got = ml.tools.execute('run_custom', {
            'device': 'cpu', 'code': NET, 'dataset_id': ds, 'name': 'u',
            'overrides': {'epochs': 3, 'batch': 4, 'val_split': vs}})
        print('   ok =', got.get('value', {}).get('ok', False),
              '| err =', (got.get('error') or {}).get('message', '')[:120])
    finally:
        ml.close()
