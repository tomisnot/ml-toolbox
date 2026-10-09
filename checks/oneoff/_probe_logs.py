# -*- coding: utf-8 -*-
"""临时探针：复现 A5 判据里 read_job 的失败，打印完整回执。"""
import io
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(r'D:\code-nosync\Mechine learning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / 'mecha'))

from ml_mecha.assembly import assemble_ml_mecha          # noqa: E402
from ml_mecha import job_logs                            # noqa: E402

NET = '''
class Network:
    def build(self, n_in, n_out, *, nn):
        print("[net] build ok \\u26a0 non-ascii")
        import sys
        print("[net] stderr line", file=sys.stderr)
        return nn.Sequential(nn.Linear(n_in, 4), nn.ReLU(), nn.Linear(4, n_out))
'''

td = tempfile.mkdtemp(prefix='logs-')
app = assemble_ml_mecha(root=td, seed=13)
try:
    sys.path.insert(0, str(REPO / 'tests'))
    import test_ml_mecha as T                              # 复用它的 _prepared
    dsid = T._prepared(app)
    print("dsid =", dsid)

    real_out, real_err = sys.stdout, sys.stderr
    spy = io.StringIO()
    job_logs.uninstall()
    sys.stdout = spy
    try:
        job_logs.install()
        job = app.submit({"command": "run_custom", "code": NET, "dataset_id": dsid,
                          "name": "logs", "overrides": {"epochs": 2, "batch": 4}},
                         side="ai")
        status = app.wait_job(job, timeout=180.0)
        print("job state =", status.get("state"), "| error =", status.get("error"),
              "| result =", json.dumps(status.get("result"), ensure_ascii=True,
                                       default=str)[:400])
        got = app.tools.execute("read_job", {"job_id": job.id, "logs": 50})
        print("read_job =", json.dumps(got, ensure_ascii=True, default=str)[:1200])
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        job_logs.uninstall()
    print("spy 收到 =", json.dumps(spy.getvalue()[:300], ensure_ascii=True))
finally:
    app.close()
