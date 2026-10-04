---
name: ml-toolbox
description: >-
  通过 MCP 操控 ML Toolbox（机器学习工具箱：59 个经典方法 + 自动调参 + 神经网络）——
  选数据、组装预处理链、跑方法或自定义网络、读训练过程观测、对比与后台长训练。
  含工具清单、正典工作流、长训练姿势、图任务（GNN）与当前能力边界。
whenToUse: >-
  当需要做机器学习实验（准备数据、选方法、跑训练、比指标、看训练曲线、写自定义网络、
  跑图神经网络）时。软件需处于 **AI 模式**（人经 `python launcher.py` 选「AI 模式」）。
---

# SKILL: ML Toolbox — 机器学习工具箱（AI 操作指南）

> 本文件是 AI harness 的 skill 说明书：连接方式、工具清单、正典工作流、纪律与已知边界。
> **AI 使用中发现文档与实态不符，应回头修正本文件**（自我迭代）。
> 工具清单那张表由 `tests/test_skill_surface.py` 与**真注册表**逐名对账
> （清单少一个/多一个都判红），所以**改工具面必须同时改这张表**。

---

## 什么时候用

- 要在一份数据上**试方法、比指标**（分类/回归/聚类/降维/异常/时序）；
- 要**训练神经网络**，或写一个**自定义结构**（残差、多分支、图卷积…）；
- 要看**训练过程**（loss/val_loss/grad_norm/lr/dead_relu/逐层梯度）而不是只看最终分数；
- 要**长训练**（跑得比一次工具调用还久）；
- 要做**图任务**（节点分类）。

## 什么时候**不**适用（先看清，免得白跑）

- **软件没在 AI 模式**：写操作会被 `authority_locked` 挡住。这不是故障——写权出厂锁死，
  **只有人类能开闸**（AI 无法自解锁，这是刻意的安全边界）。
- **链路预测 / 图分类**：本工具支持的是**节点分类**式的图任务；"样本是边对"或
  "一行=一整张图"这两类**不在能力面内**（标签的单位不是"行"）。别硬试。
- **要拿回某个 epoch 的权重**（`keep_best`）：**没做**。今天只能查到 `best_epoch` 是第几轮。
- **对新数据预测 / 导出预测**（`predict_new` / `export_prediction`）：**工具面没有接线**
  （GUI 有）。别以为能调。
- **自动调参**（`tune_method`）：MCP 工具面**没有**（GUI 有）。

---

## 工具清单

> 共 **17** 个。表里的名字必须与注册表逐字一致（有判据钉着）。

| 工具 | 干什么 | 长什么样的大结果 |
|---|---|---|
| `describe_methods` | 列可用方法（59 个） | 名字/族/任务，不列参数 |
| `describe_method` | 某个方法的可调参数与取值范围 | 参数表 |
| `describe_dataset` | 已准备数据的规模/指纹/预处理链逐步参数/实际划分口径 | 摘要，不内联数据 |
| `describe_run` | 按 `run_id` 回看一次运行 | 指标 + 产物类别名（数值不内联） |
| `describe_training` | **训练过程观测**：逐 epoch 的 loss/val_loss/grad_norm/lr/dead_relu + 逐层梯度 | 默认只给**最新一轮**；`include_series` 才给序列；可用 `pick` 只取几条 |
| `describe_networks` | 列出自定义网络（名字/指标/源码行数/用过几次） | 清单，不含源码 |
| `describe_network` | 按名字取回网络**源码**（改一版再跑，免重贴） | 源码 + 来源 run |
| `prepare_dataset` | 登记数据 + 拟合预处理链；可**自己组装**这条链（选步骤/换顺序/写自定义步骤/指定切分/给图边表） | `dataset_id` + 链摘要 + 划分口径 |
| `run_method` | 跑一个内置方法（**同步**，适合短任务） | `run_id` + 指标 |
| `run_method_batch` | 顺序跑多个方法 | 每个方法的 `run_id` + 逐项失败原因 |
| `compare_methods` | 多方法对比 | 对比表 + 最优（**要比多个就用它**，别逐个拉） |
| `run_custom` | 跑一段**自定义网络代码**（AI 写结构，系统管训练与观测） | `run_id` + 指标 |
| `submit_run` | 把一次运行提交为**后台任务**（给 `method` 或 `code`）——**长训练就走这条** | `job_id`（立刻返回） |
| `read_job` | 查后台任务进度/结果 | 状态 + **逐代 loss/val_loss** + **中途就能拿到的 `run_id`** |
| `cancel_run` | 请求取消后台任务 | 明确回"已请求"（在 epoch 边界生效，不谎报"已取消"） |
| `run_queries` | **一次往返跑一串只读查询**（列方法→看参数→查数据集→看训练） | 每步自带结果与短摘要 |
| `clear_current_state` | 把配置台面清零（数据集引用/方法/覆写/种子/设备） | 逐键"清空前值 + 清空后读回值" |

---

## 标准工作流（照做即可）

```
① describe_methods                       # 有哪些方法（可按 family 过滤）
② describe_method(name=…)                # 拿到**真实**的参数名与范围（别猜）
③ prepare_dataset(source=…, target=…)    # 登记数据；要自己组装链就给 pipeline=
④ run_method(method=…, overrides={…})    # 短任务；长任务改 submit_run
⑤ describe_run(run_id=…)                 # 指标与生效参数
⑥ describe_training(run_id=…)            # 过程数值（判断该不该早停/是不是爆了）
```

**要比多个方法**：直接 `compare_methods`（它给对比表），不要 `describe_run` 拉 N 次再自己对齐。

**要在干净台面上开始**：先 `clear_current_state`（否则会踩到上一轮留下的数据集/方法/覆写）。

---

## 长训练：只能用"发信号 + 主动查"

LLM 的一次工具调用本质是 HTTP 请求，**客户端必须有超时**（本实例 10 分钟）
⇒ "同步等训练跑完"对长训练**天生不成立**。正确姿势：

```
submit_run(method=… 或 code=…)  → job_id      # 发完就走，不等
read_job(job_id)                → state + 逐代 loss/val_loss + run_id
describe_training(run_id)       → 想查哪一代/哪个指标都行
cancel_run(job_id)              → 在 epoch 边界停下（协作式）
```

- 后台任务**终态自动落盘** ⇒ 就算后来丢了 `job_id`，结果还在 `runs/<run_id>/`。
- **并发上限是 4**（超额会回 `too_many_jobs`，不排队）：要跑更多先等前一批结束。
- 提交前会**同步预检写权**：LOCKED 下当场拒（不会"提交成功再发现失败"）。

---

## ⭐ 三条省回合/省 token 的纪律

1. **`run_queries`**：一串**只读**查询一次往返跑完（"列方法→看三个方法的参数→查数据集"）。
   ⚠ 它**只读**：写操作（准备数据、跑方法、改配置）仍要单独调用。
2. **`pick`**：`describe_training(include_series=true, pick=["series.loss.values"])`
   只回关心的那几条（实测省 ~65%）。取不到的键会如实列进 `_missing`（不会静默回 null）。
   ⚠ **回执里声明为必填的键删不掉**（工具面契约），所以别指望 `pick` 把回执变空。
3. **`overrides` 只写变化量**：只列要改的键（不用抄整份参数表）。
   ⚠ **参数名先查 `describe_method`**：拼错会被拒（不会静默忽略），但那是白跑一轮。

---

## 写权与失败：两条必读

- **写权**：出厂 `LOCKED`，只有人类能开闸。被拒时 `error.info.kind` 是
  `authority_locked` 或 `authority_mode_mismatch`——**这不是 bug，别重试**，请人类开闸。
- **失败是可教学的**：读 `error.info.kind`（机读分类）+ `hint`/`suggest`（怎么改）。
  常见：`bad_param_value`（参数值不对，hint 会给合法范围）、`unknown_key`（名字不存在）、
  `bad_graph_edge`（边表不是数字）、`unknown_dataset`（hint 会列出本会话可用的 id）。
  **按 hint 改，别盲试。**

---

## 图任务（GNN）：能做什么、不能做什么

能做的：**节点分类**（一行=一个节点、一个标签）。

```python
# 建数据：节点特征 + 边表（行号指向观测表的行）
prepare_dataset(source={"kind": "arrays", "X": ..., "y": ...}, target="target",
                graph={"edges": [[0, 1], [1, 2]], "directed": False},
                pipeline={"steps": [{"key": "scale"}]})

# 跑图网络：run_custom 的代码多写一个 forward_graph
class Network:
    def build(self, n_in, n_out, *, nn):
        return MyGCN(n_in, 16, n_out)

    def forward_graph(self, model, ctx):
        return model(ctx["x"], ctx["edge_index"], ctx["n_nodes"],
                     ctx.get("edge_weight"))
```

- `ctx` 里有：`x`、`edge_index`(2×E)、`edge_weight`、`n_nodes`、`train_mask`/`test_mask`、`torch`、`nn`。
- **划分走节点掩码**（同一张图、只对指定节点算损失）——**不要**想着按行切分。
- **图任务不走交叉验证**（CV 按行切会静默失真 ⇒ 会被明确拒绝）。
- **图模型不能对"一批新行"预测**（前向需要整张图的边）；测试节点的预测随运行产出。

---

## 自定义网络：契约与复用

```python
class Network:
    def build(self, n_in, n_out, *, nn): ...        # 必需：返回 nn.Module
    def loss(self, *, nn, n_out): ...               # 可选：默认按任务给 CE/MSE
    def configure(self, model, *, torch, lr, weight_decay): ...   # 可选：自定优化器
    def forward_step(self, model, batch, ctx): ...  # 可选：表格任务的特殊批次
    def forward_graph(self, model, ctx): ...        # 可选：图任务的全图前向
```

- 代码**只写结构**：不许自己写盘、不许碰 GUI（观测与存档由系统统一负责）。
- **按名字复用**：`run_custom(name="resnet18", persist=true)` 存下来之后，
  `describe_networks` 列表、`describe_network(name=…)` 取回源码改一版再跑 ⇒ **别重贴代码**。
- ⚠ `run_id` 是**确定性**的（方法名+毫秒+种子）：同一 `seed` 重跑会**复用同一个
  `runs/<run_id>/tb/`**（事件会追加），需要干净曲线就换个 `seed`。

---

## 预处理链：自己组装（可选）

默认链：中位数填 → one-hot → 标准化 → 8:2 切分。要换策略/加特征工程/改切分：

```jsonc
{"steps": [{"key":"missing","params":{"strategy":"mean"}},
           {"key":"ratio","code":"class Step(StepBase):\n …"},   // 自定义步骤
           {"key":"scale","params":{"mode":"robust"}},
           {"enabled": false, "key":"feature_select"}],
 "split": {"test_size": 0.2, "stratify": true, "seed": 7}}
```

- **顺序就是你给的顺序**（不重排、不去重）；`enabled:false` 跳过该步。
- **坏声明会报错**（未知步骤/未知键/坏取值），**不静默忽略**。
- 要按**训练集统计量**变换的步骤（均值/分位数）**必须自己写 `transform`**，
  否则新数据会被按它自己的统计量变换（静默的错）。

---

## 已知边界（如实登记，别当 bug 报）

| 事项 | 现状 |
|---|---|
| 链路预测 / 图分类 | **不支持**（标签单位不是"行"） |
| `keep_best`（取回最佳轮次权重） | **未做**；只能查 `best_epoch` |
| `predict_new` / `export_prediction` | 工具面**没接线**（GUI 有）；会话层有实现 |
| `tune_method`（自动调参入口） | 工具面**没有**（GUI 有） |
| 图任务 × 交叉验证 | **明确拒绝**（会静默失真） |
| 后台任务并发 | 上限 **4**（不排队） |
| 设备（`device`） | 真透传到方法层（`auto`/`cpu`/`cuda`） |
