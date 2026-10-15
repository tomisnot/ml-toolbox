# ML Toolbox · 机器学习方法试验台

把“哪个方法适合这份数据”变成可比较、可检视、可复现的实验过程：共享数据与处理管道，尝试不同方法，观察训练和结果，再继续调参。

ML Toolbox 同时提供可直接 import 的 Python API、PyQt5 桌面软件，以及经 mecha 接入的 AI 工具与观测面。方法、数据和实验属于软件；AI 是选择、编写模型和组织实验的操作者。

## 主要能力

| 能力 | 内容 |
|---|---|
| 方法试验台 | 回归、分类、聚类、降维、异常检测、时序、神经网络与量子启发方法；按任务和建模目的检索 |
| 共享数据管道 | 缺失值、编码、缩放、特征选择和划分；处理链可检视，新数据沿用训练变换 |
| 对比与检视 | 指标对比、方法自有图表、数据视图、核心图画廊和历史回看 |
| 自定义网络 | 表格、图与 NQS 路径；真实负载试跑、短步训练、自定义指标、checkpoint 与续训 |
| 优化试验台 | ask-and-tell 优化器、预算与约束、外部程序目标函数、自动调参和响应面 |
| 实验资产 | 运行记录、指标、工件和训练观测；通过稳定 API 与 AI 工具查询 |

方法库覆盖线性、集成、SVM、近邻、聚类、流形、时序、贝叶斯、异常检测和神经网络等家族。具体方法和参数以当前注册表为准。

## 安装

Python 3.10+。建议为项目创建独立环境：

```bash
git clone https://github.com/tomisnot/ml-toolbox.git
cd ml-toolbox
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv/Scripts/Activate.ps1`，Linux/macOS 使用 `source .venv/bin/activate`。

```bash
python -m pip install -e .
```

按需要安装可选能力：

```bash
python -m pip install -e ".[neural]"  # PyTorch 与训练事件观测
python -m pip install -e ".[optuna]"  # Optuna 优化器适配
```

AI 模式推荐同时检出当前 mecha 源码，并在本环境安装：

```bash
# 在共同父目录检出 mecha；随后回到本项目根
git clone https://github.com/tomisnot/mecha.git ../mecha
python -m pip install -e "../mecha[mcp]"
```

`.[ai]` extra 提供基于 mecha `v0.2.0` 的发布依赖；当前开发代码建议与框架源码配套使用。目录布局不同，请替换 `../mecha`。

`requirements.txt` 保留本地开发的完整安装清单，其中 mecha 使用兄弟目录的可编辑引用；使用该清单时需按实际布局放置框架仓。

## 启动

```bash
# 本地桌面软件
python app.py

# 本软件的 headless MCP 服务
python app_entry.py --mcp

# 软件服务，同时提供可按需打开的桌面窗口与 Web 面板
python app_entry.py --mcp --with-gui --with-panel

# 由 mecha 统一启动软件、AI 宿主与项目面板
python -m mecha --all --root .
```

桌面软件有两个工作区：**ML 方法试验台**用于数据、方法对比、检视和预测；**优化调参**用于变量空间、目标函数、约束、预算与优化过程。

整站 AI 界面需要 PATH 中的 dsh 可执行命令和本仓 `dsh/` 插件资产。插件通过本地 `file:` 引用共享面板；换目录布局时，先将 `dsh/package.json` 的 `@mecha/dsh-panel` 指向所用 mecha 仓的 `mecha/dsh-panel/`，再在 `dsh/` 执行 `npm install` 和 `npm run bundle`。

## Python API

无需打开 GUI 即可准备数据、运行方法、比较记录和预测。以下示例读取含目标列 `y` 的 CSV：

```python
from ml_toolbox.api import RunRequest, Session
from ml_toolbox.core.dataset import Dataset

session = Session()
spec = session.prepare(Dataset.load("train.csv", target="y"))
record = session.run(
    spec,
    RunRequest(method="random_forest", persist=True),
)
print(record.run_id)
print(record.result.metrics)
```

`Session` 提供请求、运行状态和结果边界；方法扩展仍使用 `MLMethod`、`ParamSpec`、`PageSpec` 与注册表。数据、方法和持久化约定见 [开发者指南](docs/开发者指南.md)。

## AI 操作与训练观测

AI 工具可检索方法和参数、准备数据、运行或比较方法、编写自定义网络，并读取运行与训练结果。

- 自定义网络支持 `code_path`，便于复用源码；`dry_run` 与 `warmup_steps` 用于先检查真实负载和短步训练。
- `submit_run` 提交后台任务；`read_job` 默认返回摘要，可等待终态或多个任务中的任意一个完成。
- 训练期间可取得 `run_id`，读取曲线、指标和日志尾段；全量信息按需查询。
- 后台运行保存业务结果；自定义网络支持周期 checkpoint、声明最佳指标和从 checkpoint 续训。
- 超过并发名额的任务进入队列，排队状态和等待时间可查询。
- 写操作由 mecha 治理并留痕；人在控制面切换写权，软件无需重启。

方法试算、指标意义、资源预算和训练恢复属于本项目；通用写权、记录与连接机制由 [mecha](https://github.com/tomisnot/mecha) 提供。详细用法见 [使用指南](docs/使用指南.md)。

## 架构与扩展

```text
ml_toolbox/api.py     稳定的无界面 Python API
ml_toolbox/core/      数据、管道、契约、注册表、运行与持久化
ml_toolbox/methods/   方法实现、神经网络与绘图
ml_toolbox/opt/       优化器、目标函数、约束与 ML 桥接
ml_toolbox/ui/        PyQt5 界面与可注册检视页
ml_mecha/            受控命令、AI 工具、任务调度与观测接线
dsh/                 AI 宿主中的项目面板
tests/               按主题选择的验证
benchmarks/          方法、优化与性能实验
docs/                项目定位、契约与使用说明
```

计算核心与 GUI 分开，方法通过声明提供参数和检视页。新增方法注册后可进入检索、运行与对比流程；需要新的视图时可注册页面构建器。图方法通过 `Problem` 的图结构和节点掩码访问数据，避免将图任务强行变成普通表格任务。

优化侧通过 AutoTuner、响应面和代理模型与方法库互通，具体能力以运行时命令和查询声明为准。

## 开发与验证

```bash
python tests/run_all.py
python tests/run_all.py --only api
python tests/run_all.py --suite training
python tests/run_all.py --suite integration
python tests/run_all.py --all
```

默认运行核心；训练、Qt 界面、面板资产、优化、性能实验和发布扫描可分别选择。`--all` 是常规软件全量，不包含独立的 benchmarks 与 release 套件。完整命令和环境依赖见 [验证说明](tests/README.md)。

日常迭代运行受影响项，跨层变化再扩大范围。文档和工具数量不构成测试的第二份注册表，普通 pytest 模块由实际收集机制执行。修改面板时在 `dsh/` 运行 `npm test`。

## 当前边界

- 内置神经网络方法以 MLP 为主；其他结构通过自定义网络路径接入，支持程度取决于数据与训练契约。
- 自定义神经网络可通过 checkpoint 续训；普通 estimator 和优化器不承诺通用的精确中断恢复。
- 后台 `job_id` 属于当前进程，重启后应通过保存的 `run_id` 和业务存档查结果。
- 量子启发方法是经典态矢量或密度矩阵模拟，不宣称量子硬件加速。
- 部分时序方法只消费目标序列，外生特征支持需逐方法确认。
- 核方法、大样本流形方法和大模型训练有各自的计算成本。Windows 是主要验证环境。

## 文档与许可

- [使用指南](docs/使用指南.md)
- [开发者指南](docs/开发者指南.md)
- [历史文档](docs/history/)

许可：[MIT](LICENSE)。
