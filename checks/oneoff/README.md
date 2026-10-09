# `checks/oneoff/` —— 一次性调查脚本（**不是判据、不被任何代码 import**）

这里放的是某几轮排查/验收时写下的**一次性脚本**，用途是"把一次现象钉成可复现的读数"。
它们**不在门禁里**、**没有任何模块 import 它们**（加进来之前扫过 `*.py`），
所以移动/删除都不影响 `tests/run_all.py` 或任何一个判据。

## 两类

| 前缀 | 来历 | 说明 |
|---|---|---|
| `_probe_*.py` | 2026-10-07 那一轮"接口审查"（`docs/反馈-ML工具使用体验与接口审查.md`） | 大多是**只读探针**：读 TB 事件、读 checkpoint、量回执体积。`docs/反馈-ML工具使用体验与接口审查.md` §5 附录引用了它们的旧路径（仓根），现已移到这里 |
| `_dev_*.py` | 同一时期另一条线上的 P0/P2 验证脚本（**孤儿**，作者会话已关闭） | 保留作证据；跑之前先看它开头的用法注释 |

## ⚠ 三条使用须知

1. **路径与 run_id 是写死的**（例如 `D:\code-nosync\Mechine learning`、`custom_network-2334125-0`）——
   那些 run 可能已被清掉，脚本会如实报"读不到"而不是造数（这是它们的既有纪律）。
2. 其中 `_dev_p0_e2e*.py` 需要**先设** `ML_TOOLBOX_RUNS`（见它 docstring 里的示例命令），
   否则会往仓里的 `runs/` 写 —— 那会污染实验记录。
3. 它们**不是判据**：`tests/` 下的才是。想让某条读数长期有效，请把它**固化成判据**
   （这一轮的读数已经固化在 `tests/test_nqs_path.py` / `tests/test_ai_efficiency.py` /
   `tests/test_ml_mecha.py` / `tests/test_ml_delivery.py`）。

## 唯一一个"有长期价值"的：`_factcheck_isatty_utf8.py`

它回答一个**会反复被问到**的事实：**子进程的 stdout 变成管道时，`app_entry.main()` 的
utf-8 `reconfigure` 还成立吗？** 结论：**成立（无条件，不看 `isatty`）**，
证据见 `docs/pitfalls.md` A56 的"不成立的反对理由（实测推翻）"那一段。
跑法（⚠ 必须**先剥掉** `PYTHONIOENCODING`，否则测的是"注入生效"而不是默认行为）：

```powershell
Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
python checks/oneoff/_factcheck_isatty_utf8.py
# 期望：PIPE 时 BEFORE gbk isatty=False → AFTER utf-8 isatty=False；不调 main 的对照是 gbk
```
