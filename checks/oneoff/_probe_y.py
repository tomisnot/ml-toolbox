# -*- coding: utf-8 -*-
"""临时探针：prepare_dataset 之后 spec.y 到底是什么？（查 Long target 的根因）"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402

td = tempfile.mkdtemp(prefix='y-')
ml = assemble_ml_mecha(root=td, seed=7)
ml.switch_ai()
try:
    rng = np.random.RandomState(3)
    frame = pd.DataFrame(rng.randn(24, 4), columns=list('abcd'))
    frame['y'] = frame['a'] - 0.5 * frame['b'] + 0.1 * rng.randn(24)
    print('原始 frame.shape =', frame.shape, '| y dtype', frame['y'].dtype)
    got = ml.tools.execute('prepare_dataset', {
        'source': {'kind': 'frame', 'frame': frame}, 'name': 'u', 'target': 'y',
        'pipeline': {'steps': [{'key': 'scale'}]}})['value']
    print('回执 =', {k: got[k] for k in got if k in
                    ('dataset_id', 'rows', 'cols', 'target', 'target_kind',
                     'split_policy')})
    entry = ml.engine.dataset_entry(got['dataset_id'])
    spec = entry.spec
    print('spec.X.shape =', spec.X.shape, '| spec.y dtype =', spec.y.dtype,
          '| target_kind =', spec.target_kind)
    print('spec.y.head =', list(spec.y.head(5)))
    print('spec.X.columns =', list(spec.X.columns))
    print('chain =', [(s.get('key'), {k: v for k, v in s.items() if k != 'key'})
                      for s in (spec.meta.get('chain') or [])][:6])
finally:
    ml.close()
