# 验证的运行方式

```powershell
python tests/run_all.py                      # 核心契约与小数据结果
python tests/run_all.py --only api            # 按名称选，不受默认档位限制
python tests/run_all.py --suite integration   # mecha、真实协议与交付链
python tests/run_all.py --suite training      # 神经网络、图/NQS 与方法库冒烟
python tests/run_all.py --suite ui            # offscreen Qt
python tests/run_all.py --suite assets        # 类型、真实面板构建、附加页和绘图结果
python tests/run_all.py --suite optimization  # 优化算法与 runner
python tests/run_all.py --all                 # 常规全量
python tests/run_all.py --suite benchmarks    # 性能实验
python tests/run_all.py --suite release       # 发布前样例路径审查
python tests/run_all.py --all --list          # 查看范围
```

套件串行执行；原来的“图/范式串行”说明与 Popen 全并行实现不符，且并发训练曾耗尽页面文件。每项显示实际耗时，失败打印完整输出，--timeout 设置单项预算。更长训练参数仍可用 MLTB_SLOW=1 选择，不属于日常验证。

pytest 统一收集普通测试并提供夹具；冒烟、优化、旧回归/UI 和 Node 包装保留兼容脚本入口。移除检查所有测试源码是否登记的元守卫，由实际收集机制解决漏跑。核心需现有 ML 依赖，集成需 mecha/MCP，界面需 Qt，资产需 Node；具体缺失由对应验证报告。

退役技能文档登记/词汇/清单镜像门禁与跨仓领域词扫描。项目边界通过真实依赖和公共行为评估，不禁止不同项目使用相同通用名称。公共参数的 nullable 行为仍逐个验证，不固定可选参数总数；能力图的真实投影测试保留，文档快照可按需生成。

run_id 的检查移到临时目录，显式 id 经公共 Session API 实际运行验证。删掉原先没传 fixed id 的重复神经训练和检查 if 分支源码的断言。

测试验证结构与可观察结果；文档记录理由和常用工作流，不要求固定章节、完整工具表或每条配红证。

浏览器依赖边界交给已有 platform=browser 构建器，删除自制 import 闭包解析器。资产入口执行 TypeScript、真实浏览器构建和项目附加页行为测试各一次，不再为测试数设置下限或重复跑 Node 检查。

样例个人路径扫描移到发布套件，不固定样例文件总数，不再随每轮训练回归运行。

本次验证：日常核心 4 个套件通过（13.8s），集成 12 个套件通过（272.3s）；训练、优化、UI、绘图与面板已分别执行。早期旧入口的 fixture/登记问题已修复并复验；资产类型、构建和附加页行为通过。发布扫描 1 例通过。

新增普通 test_*.py 模块由 --all 自动收集，--only 也可直接筛选；不要求先登记才会执行。套件名单仅用于按主题选择，以及明确分离性能和发布场景。
