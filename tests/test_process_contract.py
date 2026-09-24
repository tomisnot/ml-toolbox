# -*- coding: utf-8 -*-
"""External process contract tests that do not require the GUI."""
from __future__ import annotations

import os
import sys

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


def main():
    tests = [test_process_env_extra_reaches_child]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
