"""按改动范围运行 ML 回归。默认 core；训练、集成、界面和性能单独选择。
--all 包含普通软件回归，不含 benchmarks/release。串行执行，避免训练争用和页面文件耗尽。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    'core': ('test_api', 'test_resources', 'test_pipeline_safety', 'test_fitted_pipeline'),
    'integration': ('test_process_contract', 'test_ml_capabilities', 'test_ml_mecha',
                    'test_ml_delivery', 'test_tensorboard', 'test_panel_web',
                    'test_authority_read', 'test_control_endpoint', 'test_manual_surface',
                    'test_mecha_site', 'test_ml_residual_architecture', 'test_runs_limit'),
    'training': ('test_regressions', 'test_problem', 'test_nqs_path', 'test_ai_efficiency', 'smoke_test'),
    'ui': ('test_ui',),
    'assets': ('test_dsh_panel', 'test_plot_spec'),
    'optimization': ('test_opt',),
    'benchmarks': ('test_nn_perf',),
    'release': ('test_release',),
}
SOFTWARE_SUITES = tuple(k for k in SUITES if k not in ('benchmarks', 'release'))
LEGACY_SCRIPTS = {'smoke_test', 'test_opt', 'test_regressions', 'test_ui', 'test_dsh_panel'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='append', choices=SUITES)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--only', action='append')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout 必须大于 0')
    suites = (SOFTWARE_SUITES if args.all else args.suite
              or (tuple(SUITES) if args.only else ('core',)))
    selected = list(dict.fromkeys(name for suite in suites for name in SUITES[suite]))
    # New ordinary modules are discovered for --all/--only without an admission registry.
    if args.all or args.only and not args.suite:
        classified = {name for names in SUITES.values() for name in names}
        selected += sorted(p.stem for p in (ROOT / 'tests').glob('test_*.py')
                           if p.stem not in classified)
    if args.only:
        for pattern in args.only:
            if not any(pattern in name for name in selected):
                parser.error(f'没有匹配的验证项：{pattern}')
        selected = [name for name in selected if any(p in name for p in args.only)]
    if args.list:
        print('\n'.join(f'tests/{name}.py' for name in selected))
        return 0
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'QT_QPA_PLATFORM': 'offscreen'}
    failed = 0
    start = time.monotonic()
    for name in selected:
        path = ROOT / 'tests' / f'{name}.py'
        # pytest owns collection and fixtures; retain direct entry only for legacy script suites.
        direct = name in LEGACY_SCRIPTS
        cmd = ([sys.executable, '-X', 'utf8', '-B', str(path)] if direct else
               [sys.executable, '-X', 'utf8', '-B', '-m', 'pytest', '-q', str(path)])
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=args.timeout)
            output, ok = (r.stdout + '\n' + r.stderr).strip(), r.returncode == 0
        except subprocess.TimeoutExpired:
            output, ok = f'超过单项预算 {args.timeout:g}s', False
        failed += int(not ok)
        last = output.splitlines()[-1] if output else '无输出'
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.monotonic()-t0:.1f}s) — {last}", flush=True)
        if not ok:
            print(output, flush=True)
    print(f'run_all: {len(selected)-failed} PASS / {failed} FAIL — {time.monotonic()-start:.1f}s')
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())
