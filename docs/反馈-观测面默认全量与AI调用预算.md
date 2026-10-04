# 反馈：观测面的"默认全量"与 AI 的调用预算（含实测证据）

> **上一份**（`建议-AI往返成本与训练观测面.md`）提的 5 条**已经落地得很好**，
> 本文只记**这一轮新撞到的**，并明确"已经好用的部分"（§1），免得被当成整体否定。
> 依据是**一次完整训练**（3 个后台任务对照 + 3 次 `dry_run` + 离线分析）。

---

## 1. 先说已经好用的（这轮真省了往返）

| 功能 | 实测 |
|---|---|
| **`code_path`** | ✅ 3 个对照任务各 150 行代码，只报路径；回执 `code_source: "file"`、`code_lines: 154` 说清读的是哪一份。**这条我上次报的缺口被完整修好了** |
| **`dry_run`** | ✅ 这次它一次就**指认了出错的模块**（"失败在 `to_tab`（进去了、没出来）：它收到 [...]"），0.1 秒定位。三个任务从"写错"到"跑通"只花了 3 次 `dry_run` |
| **`on_step` 返回 dict** | ✅ 步级 `fid/step` 真的进了 tags 与观测面 |
| **`metrics` 带 `eval+no_grad` + `metrics_every`** | ✅ 无异常，贵指标降频可用 |
| **`build(…, ctx=…)`** | ✅ 从 `payload_shapes` 取维度，不用写死 |
| **失败的结构化归因** | ✅ `error_kind` / `error_hint` 准确（`shape_mismatch`、`missing_network_class`、`bad_code_source` 都对得上） |

---

## 2. 缺口 A【最高】`read_job` 的默认回执**太长**，导致我"放弃官方轮询姿势"

### 2.1 实测

`read_job` 的 `output.required` 里有 **`progress`** 与 **`epochs`**：

```
read_job.required = ok, job_id, state, cancel_supported,
                    progress, progress_latest, run_id, epochs, result_ref, error, ref_error
```

而这两个键装的是**全量历史**：

- `progress`：每轮一条 `{frac, stage}` ⇒ 1200 epoch 就是 1200 条
- `epochs`：每轮一条 `{epoch, loss, val_loss}` ⇒ 又是 1200 条

**后果**：我为了看一次进度，拿到几百 KB 的文本。**这不是抱怨 token，是抱怨我因此改了行为**——
我在回复里写过："`read_job` 的 `progress` 数组太长，消耗很大——后面我不再直接轮询它了。"
**AI 主动放弃官方推荐的长训练姿势，是这条缺口最硬的证据。**

### 2.2 建议

```jsonc
// 默认给"摘要"：state + progress_latest + run_id + 最近 N 轮
read_job(job_id="job-1")                     // 几行
read_job(job_id="job-1", detail="full")      // 才给全量
read_job(job_id="job-1", tail=20)            // 或显式要最近多少轮
```

关键点：**"最新状态"与"全量历史"是两个不同的需求**，现在被绑在一个必填键里。
`progress_latest` 已经有摘要信息（state/frac/loss/val_loss）——把它做成默认就够了。

---

## 3. 缺口 B【高】`pick` 省不动必填键里的"大块"

### 3.1 实测

我想只看 4 个指标，用了：

```jsonc
describe_training(run_id=…, pick=["latest.other"])
```

回执里 `latest`（**必填**）仍带着 **30 个 `grad_by_layer`**：

```jsonc
"latest": {"loss":…, "val_loss":…, "grad_norm":…, "lr":…, "step":…,
           "grad_by_layer": {"pe":…, "feat.weight":…, … 共 30 项},
           "other": {…30 项梯度…}}
```

`pick` 省掉了别的键，但**省不掉必填键内部的大块**。于是"只看 4 个数"要付"30 个梯度"的钱。

### 3.2 建议

- `latest.grad_by_layer` 改成**默认摘要**（前 3 + 后 3 + 总数），要全量给 `grad_layers="full"`；
- 或者允许 `pick` **下钻到必填键内部**（现在是"必填键删不掉"，但那不等于"内部不能再筛"）。

---

## 4. 缺口 C【中高】`dry_run` 只验前向 ⇒ 抓不到"训练动态"类错误

### 4.1 实测：这轮最贵的坑，`dry_run` 看不见

这轮有**三次长训练全部分散/发散**（取消了两个）：

| run | 现象 | `dry_run` 能不能抓到 |
|---|---|---|
| 双重加权（`self.fw` × 20 = 400） | step 120 处 loss 从 **0.009 → 113** | ❌ 只跑前向 |
| 同配置去掉双加权 | 仍涨到 **71** | ❌ |
| 旧 NLL 目标 | `nll → 0.0000`、`最大概率=1.000000`（退化解） | ❌ |

`dry_run` 在形状/vtable 层**非常准**，但它**不反传**，所以：
权重尺度是否病态、loss 是否在几十步内爆掉、目标函数是否存在退化解——**全在它的盲区**。
而这恰好是长训练里最贵的错误（一次 1200 epoch 的时间 + 一次取消 + 我读几百 KB 日志）。

### 4.2 建议

```jsonc
dry_run=true, warmup_steps=20      // 真**反传** N 步（默认 0 = 现在的前向试跑）
```

回执加：`loss_curve`（N 个点）、`grad_norm_curve`、`params_changed`、`any_nonfinite`。

边界不变（**不落盘、不计入记录**），成本是几十步而不是上千步。
判据很清楚：**N 步内 loss 涨一个数量级 ⇒ 这是"你的配置会发散"，不是"框架坏了"**。

---

## 5. 缺口 D【中】重启会清空会话态，与"磁盘上有存档"的承诺脱节

`submit_run` 的文档说：

> 后台跑完会**自动落盘**（`runs/<run_id>/record.json`）——所以就算你后来丢了 `job_id`，结果也还在。

**实测**：重启（按 `SKILL.md` 的建议刷新工具表）之后：

- `read_job(job_id="job-1")` ⇒ `job 'job-1' 不存在`（`unknown_job`）
- 所有 `dataset_id` 也失效，必须重新 `prepare_dataset`

⇒ "结果还在"是**盘上还在**，但**会话里找不回来**——得靠我记住 `run_id` 才能 `describe_run`。
**建议**：`read_job` 对未知 job 时，回一句"本会话无此 job；盘上的运行可用
`describe_run(run_id=…)` 查，最近 N 个是 …"（给个线索，而不是只回"不存在"）。

---

## 6. 一句话总结

工具的能力面这轮**明显够用**（`dry_run` / `code_path` / 步级指标都真的省了往返）。
剩下的是**"默认给多少"的取向问题**：

| 缺口 | 本质 | 建议取向 |
|---|---|---|
| A `read_job` | 最新状态与全量历史绑在一个必填键 | **默认摘要**，要全量显式要 |
| B `pick` | 必填键内部的大块筛不掉 | 逐层梯度默认摘要 |
| C `dry_run` | 只验前向 ⇒ 抓不到动态类错误 | 加 `warmup_steps` |
| D 重启 | 盘上有、会话里找不回 | 未知 job 时给盘上线索 |

A 和 B 是**同一件事**：观测面默认给"全量"，而 AI 的一次调用是有预算的。
