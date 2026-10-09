# -*- coding: utf-8 -*-
"""临时探针：spy `prepare_supervised` 的输入 y（到底哪一步把它变成分类码）。"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402
from ml_toolbox.methods.neural import trainer as T        # noqa: E402
from ml_toolbox.core.pipeline import _infer_kind          # noqa: E402

src = open(REPO / 'tests/test_nqs_path.py', encoding='utf-8').read()
NET = src.split("_NET_PERIODIC = '''")[1].split("'''")[0]
_orig = T.prepare_supervised


def spy(X, y, *, kind, seed, val_split):
    print('   [prepare_supervised] kind=%s | y type=%s dtype=%s | n=%s uniq=%s'
          % (kind, type(y).__name__, getattr(y, 'dtype', None),
             None if y is None else len(y),
             None if y is None else pd.Series(y).nunique()))
    print('   [prepare_supervised] _infer_kind(y) =',
          None if y is None else _infer_kind(pd.Series(y)))
    return _orig(X, y, kind=kind, seed=seed, val_split=val_split)


T.prepare_supervised = spy
import ml_toolbox.methods.neural.custom as C               # noqa: E402
C._trainer.prepare_supervised = spy

for rows, seed in ((24, 3), (96, 0)):
    td = tempfile.mkdtemp(prefix='spy-')
    ml = assemble_ml_mecha(root=td, seed=7)
    ml.switch_ai()
    try:
        rng = np.random.RandomState(seed)
        frame = pd.DataFrame(rng.randn(rows, 4), columns=list('abcd'))
        frame['y'] = frame['a'] - 0.5 * frame['b'] + 0.1 * rng.randn(rows)
        got = ml.tools.execute('prepare_dataset', {
            'source': {'kind': 'frame', 'frame': frame}, 'name': 'u', 'target': 'y',
            'pipeline': {'steps': [{'key': 'scale'}]}})
        ds = got['value']['dataset_id']
        print('=== rows=%d seed=%d ds=%s' % (rows, seed, ds))
        r = ml.tools.execute('run_custom', {
            'device': 'cpu', 'code': NET, 'dataset_id': ds, 'name': 'u',
            'overrides': {'epochs': 2, 'batch': 4, 'val_split': 0.25}})
        print('   run ok =', bool(r.get('value', {}).get('ok')))
    finally:
        ml.close()
