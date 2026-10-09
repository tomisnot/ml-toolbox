# -*- coding: utf-8 -*-
"""临时探针：在**独立临时根**上验证 checkpoint_every（不碰正在跑的权威进程）。"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402

NET = '''
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.f = nn.Linear(n_in, 8)
        self.h = nn.Linear(8, 1)

    def forward(self, x):
        return self.h(torch.tanh(self.f(x)))


class Network:
    primary_metric = "my/step"

    def build(self, n_in, n_out, *, nn, ctx=None):
        return Net(n_in)

    def loss(self, *, nn, n_out):
        return nn.MSELoss()

    def on_step(self, model, step, ctx):
        return {"my/step": float(step)}

    def forward_step(self, model, batch, ctx):
        x, y = batch
        return model(x).squeeze(-1), y
'''

td = tempfile.mkdtemp(prefix='ckprobe-')
ml = assemble_ml_mecha(root=td, seed=7)
ml.switch_ai()
try:
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(rng.randn(40, 3), columns=['a', 'b', 'c'])
    frame['y'] = frame['a'] * 2 - frame['b']
    ds = ml.tools.execute("prepare_dataset", {
        "source": {"kind": "frame", "frame": frame}, "name": "probe",
        "target": "y",
        "pipeline": {"steps": [{"key": "scale"}]}})["value"]["dataset_id"]
    out = ml.tools.execute("run_custom", {
        "code": NET, "dataset_id": ds, "name": "ckprobe", "persist": True,
        "checkpoint_every": 3,
        "overrides": {"epochs": 6, "batch": 4, "val_split": 0.25}})["value"]
    print('run_custom 回执 =', json.dumps(
        {k: out[k] for k in ('ok', 'run_id', 'checkpoint_every', 'warnings')},
        ensure_ascii=True))
    rid = out['run_id']

    from ml_toolbox.core.persistence import RUNS_DIR
    ckdir = Path(RUNS_DIR) / rid / 'checkpoint'
    print('checkpoint 目录内容 =', sorted(p.name for p in ckdir.iterdir()))

    ck = ml.tools.execute("checkpoint_detail", {"run_id": rid})["value"]
    print('tags =', ck['tags'])
    print('last_saved =', json.dumps(ck['last_saved'], ensure_ascii=True))
    print('load_hint.state_dict_key =', ck['load_hint']['state_dict_key'])
    print('files_meta.last =', json.dumps(
        {k: v for k, v in ck['files_meta']['last'].items() if k != 'has_bias'},
        ensure_ascii=True))
    print('files_meta.best =', json.dumps(
        {k: v for k, v in ck['files_meta']['best'].items() if k != 'has_bias'},
        ensure_ascii=True))

    # 观测面：两族 tag 的轴单位
    tr = ml.tools.execute("describe_training", {"run_id": rid})["value"]
    print('step_units =', json.dumps(tr['step_units'], ensure_ascii=True))
    print('axes =', json.dumps({k: v for k, v in tr['axes'].items() if k != 'note'},
                              ensure_ascii=True))
    tr2 = ml.tools.execute("describe_training",
                           {"run_id": rid, "tags": ["loss", "my/step"]})["value"]
    print('跨族 axis_warning? =',
          bool((tr2['compare'].get('axis_warning') or '')),
          '| step_units =', json.dumps(tr2['compare'].get('step_units'), ensure_ascii=True))

    # 旧存档（本改动之前的）也要能读：拿盘上现成的那个
    old = ml.tools.execute("checkpoint_detail",
                           {"run_id": "custom_network-1202787-0"})["value"]
    print('旧存档 files_meta.best =', json.dumps(
        {k: v for k, v in old['files_meta']['best'].items() if k != 'has_bias'},
        ensure_ascii=True)[:400])
    print('旧存档 last_saved =', json.dumps(old['last_saved'], ensure_ascii=True))

    # 原子写：不该留 .tmp-*
    print('残留临时文件 =', [p.name for p in ckdir.iterdir() if '.tmp-' in p.name])
finally:
    ml.close()
    print('临时根 =', td)
