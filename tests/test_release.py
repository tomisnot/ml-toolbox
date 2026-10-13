"""发布前样例路径扫描；只检查本机路径，不禁止项目词汇或固定文件总数。"""
from pathlib import Path
import re


def _normalize(value):
    return re.sub('/+', '/', value.replace(chr(92), '/')).casefold()


def test_cases_have_no_personal_paths():
    root = Path(__file__).resolve().parents[1]
    private = (_normalize(str(Path.home())), _normalize(str(root)))
    hits = []
    for path in sorted(p for p in (root / 'cases').rglob('*') if p.is_file()):
        text = _normalize(path.read_text(encoding='utf-8', errors='replace'))
        hits.extend(str(path.relative_to(root)) for prefix in private if prefix in text)
    assert not hits, '\n'.join(hits)
