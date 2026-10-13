"""项目面板：使用现有 TypeScript 编译器、浏览器构建器和实际附加页渲染测试。

共享面板行为归 mecha 资产；不再维护源码依赖解析器、文件名黑名单或用例数下限。
"""
from pathlib import Path
import shutil
import subprocess
import sys

DSH_DIR = Path(__file__).resolve().parents[1] / 'dsh'


def test_dsh_panel_contract():
    npm = shutil.which('npm.cmd') or shutil.which('npm')
    assert npm, '资产验证需要 Node/npm'
    result = subprocess.run([npm, 'test'], cwd=DSH_DIR, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_dsh_panel_contract()
    print('test_dsh_panel: PASS（类型、浏览器构建与附加页行为）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
