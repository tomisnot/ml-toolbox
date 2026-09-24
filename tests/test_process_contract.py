# -*- coding: utf-8 -*-
"""External process contract tests that do not require the GUI."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_toolbox.core.contracts import ParamSpec
from ml_toolbox.opt.contracts import ParamSpace
from ml_toolbox.opt.process import ProcessObjective


def test_process_env_extra_reaches_child():
    space = ParamSpace([ParamSpec("a", "a", "number", 0.0, min=0, max=1)])
    cmd = ("python -c \"import os; "
           "print('score: ' + os.environ.get('MLTB_PROBE', 'missing'))\"")
    obj = ProcessObjective(cmd, space, name="env_probe",
                           env={"MLTB_PROBE": "7"})
    assert obj({"a": 0.0}) == 7.0


def _json_process_objective(script: str, case: str, strict: bool):
    space = ParamSpace([ParamSpec("x", "x", "number", 0.0, min=0, max=1)])
    cmd = (f'python "{script}" --case {case} --points {{points_file}} '
           f'--out {{out_file}}')
    return ProcessObjective(
        cmd, space, mode="json", score_field="C_dual", minimize=False,
        constraints=[{"field": "S_ret", "op": ">=", "value": 0.99}],
        strict_schema=strict, timeout=30, name="schema_probe")


def test_json_mode_fails_closed_on_missing_required_fields():
    with tempfile.TemporaryDirectory(prefix="mltb_proc_") as td:
        script = os.path.join(td, "blackbox.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(
                "import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--points'); "
                "p.add_argument('--out'); p.add_argument('--case'); "
                "a=p.parse_args()\n"
                "json.load(open(a.points, encoding='utf-8'))\n"
                "if a.case == 'missing_constraint':\n"
                "    rows=[{'id': 0, 'ok': True, 'C_dual': 0.5}]\n"
                "elif a.case == 'missing_ok':\n"
                "    rows=[{'id': 0, 'C_dual': 0.5, 'S_ret': 1.0}]\n"
                "else:\n"
                "    rows=[{'id': 0, 'ok': True, 'C_dual': 0.5, "
                "'S_ret': 1.0}]\n"
                "json.dump(rows, open(a.out, 'w', encoding='utf-8'))\n")

        strict = _json_process_objective(script, "valid", True)
        assert strict._run_point({"x": 0.0}, 0) == (-0.5, "ok")

        obj = _json_process_objective(script, "missing_constraint", True)
        score, status = obj._run_point({"x": 0.0}, 0)
        assert status == "failed"
        assert "S_ret" in obj.last_error

        obj = _json_process_objective(script, "missing_ok", True)
        assert obj._run_point({"x": 0.0}, 0)[1] == "failed"

        # Explicit lenient mode preserves the old compatibility behavior.
        obj = _json_process_objective(script, "missing_ok", False)
        assert obj._run_point({"x": 0.0}, 0) == (-0.5, "ok")


def main():
    tests = [test_process_env_extra_reaches_child,
             test_json_mode_fails_closed_on_missing_required_fields]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
