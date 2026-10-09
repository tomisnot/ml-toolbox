# -*- coding: utf-8 -*-
"""临时探针：复现 units 判据里的 run_custom 失败，打印完整错误。"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402

src = open(REPO / 'tests/test_nqs_path.py', encoding='utf-8').read()
NET = src.split("_NET_PERIODIC = '''")[1].split("'''")[0]

td = tempfile.mkdtemp(prefix='units-')
ml = assemble_ml_mecha(root=td, seed=7)
ml.switch_ai()
try:
    rng = np.random.RandomState(3)
    frame = pd.DataFrame(rng.randn(24, 4), columns=list('abcd'))
    frame['y'] = frame['a'] - 0.5 * frame['b'] + 0.1 * rng.randn(24)
    ds = ml.tools.execute('prepare_dataset', {
        'source': {'kind': 'frame', 'frame': frame}, 'name': 'units', 'target': 'y',
        'pipeline': {'steps': [{'key': 'scale'}]}})['value']['dataset_id']
    got = ml.tools.execute('run_custom', {
        'device': 'cpu', 'code': NET, 'dataset_id': ds, 'name': 'units',
        'overrides': {'epochs': 3, 'batch': 4, 'val_split': 0.0}})
    print(json.dumps(got, ensure_ascii=True, default=str)[:2000])
finally:
    ml.close()
