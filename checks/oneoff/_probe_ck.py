# -*- coding: utf-8 -*-
"""临时探针：
1) `load_state_dict(ck)`（ck = 整个存档 dict）到底是**报错**还是**静默**？缺失多少键？
2) `curve.direction()` 对自定义 tag（sign/agree、ph/std）判成什么？
3) `describe_training` 默认回执里 required 键占多大（pick 能省多少）？
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'D:\code-nosync\Mechine learning')

RID = sys.argv[1] if len(sys.argv) > 1 else 'custom_network-1202787-0'

print('=== 1) load_state_dict 的两种 strict 行为')
import torch
from ml_toolbox.methods.neural.custom import load_checkpoint
from ml_toolbox.core.persistence import RUNS_DIR

model, ck = load_checkpoint(RID, which='best')
print('   存档顶层键 =', sorted(ck.keys()))
sd = ck['state_dict']
print('   state_dict 张量数 =', len(sd))
# ① 默认 strict=True：拿整份存档当 state_dict
m1 = type(model)(*[]) if False else None
import ml_toolbox.methods.neural.custom as C
from ml_mecha.sandbox import call_network_build, compile_network
compiled = compile_network(str(ck.get('source') or ''))
build_ctx = dict(ck.get('build_ctx') or {})
m1 = call_network_build(compiled.instance, int(ck.get('n_in') or 0),
                        int(ck.get('n_out') or 0), torch.nn, ctx=build_ctx)
try:
    m1.load_state_dict(ck)
    print('   strict=True（默认）：**没有报错**（静默！）')
except Exception as exc:
    print('   strict=True（默认）：%s: %s' % (type(exc).__name__, str(exc)[:300]))
m2 = call_network_build(compiled.instance, int(ck.get('n_in') or 0),
                        int(ck.get('n_out') or 0), torch.nn, ctx=build_ctx)
res = m2.load_state_dict(ck, strict=False)
print('   strict=False：missing=%d unexpected=%d（**无异常**）'
      % (len(res.missing_keys), len(res.unexpected_keys)))
print('   正确写法 load_state_dict(ck["state_dict"]) 的 missing =',
      len(m2.load_state_dict(sd, strict=False).missing_keys))

print()
print('=== 2) curve.direction() 对自定义 tag')
from ml_mecha import curve
for t in ('sign/agree', 'ph/std', 'fid/sub1', 'loss', 'grad/feat.weight',
          'weights_norm/update', 'nll/amp'):
    print('   %-22s -> %s' % (t, curve.direction(t)))
sh = curve.shape([0, 1, 2, 3, 4, 5], [0.10, 0.20, 0.30, 0.45, 0.60, 0.80])
print('   一条明显在爬的曲线 shape.trend =', sh['trend'])
print('   convergence_note("sign/agree") =',
      curve.convergence_note('sign/agree', sh))
print('   convergence_note("fid/sub1")   =',
      json.dumps(curve.convergence_note('fid/sub1', sh), ensure_ascii=True)[:220])

print()
print('=== 3) describe_training 默认回执：required 键 vs 全部')
from ml_mecha.engine import MLEngine
got = MLEngine._query_training_detail(None, run_id=RID)
req = ("ok", "run_id", "available", "latest", "tags", "error", "curve",
       "convergence", "best_by", "compare")
def size(o):
    return len(json.dumps(o, ensure_ascii=True, default=str))
total = size(got)
keep = {k: got.get(k) for k in req if k in got}
print('   全部 = %d 字符；required 骨架 = %d 字符（pick 之后仍要回）'
      % (total, size(keep)))
for k in sorted(keep, key=lambda k: -size(keep[k]))[:5]:
    print('      required[%s] = %d 字符' % (k, size(keep[k])))
