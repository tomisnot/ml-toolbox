# 反馈：新一批省往返功能的一处接线缺口（AI 实测）

> **场景**：第一次用 `dry_run` / `code_path` / 步级 `on_step` 指标跑一次真实的 NQS 训练。
> **结论**：**四个功能三个好用**，`code_path` 有一个接线缺口导致它完全用不了。
> 本文只写实测到的东西，附带回执原文。

---

## 1. 先是好消息：`dry_run` 完全达到目的

它一次往返就把**所有**形状问题暴露了。实测用它试跑那份 NQS 网络（GNN + Transformer
编解码 + 复数输出头），回执关键字段：

```jsonc
{"ok": true, "dry_run": true, "n_params": 63068,
 "batch_sizes_checked": [1, 4, 16], "failed_batch_size": null,
 "shapes": [ {"module":"feat","in":[[1,8,6]],"out":[[1,8,6]]},
             {"module":"msg","in":[[1,8,8,18]],"out":[[1,8,8,24]]},
             {"module":"tok","in":[[128,8,2]],"out":[[128,8,32]]},
             {"module":"tok","in":[[96,8,2]],"out":[[96,8,32]]}, ... ]}
```

**为什么这就是我要的**：

- `tok` 收到 **128** 行与 **96** 行两批 —— 正好等于 K（全支撑采样数）与 M（保真度子集），
  **数量对得上**就说明采样索引与模型形状都对；
- 三档批量（1 / 4 / 16）全过，`failed_batch_size: null` ——
  上一轮那个"本地 B=3 全绿、真实 B=2 就错"的坑，这次**预先就被覆盖**了；
- `loss_first_batch` 也给了（0.5355），能顺手判断数值是否合理。

**上一轮同类工作花了约 15 个往返定位形状错误；这一次 1 个往返、0 个错误。**
这是本轮最大的效率改进，值得单独说一句。

---

## 2. 缺口：`code_path` 完全用不了（两个原因，任一都致命）

### 2.1 参数校验层：`code` 是必填，于是"只给 `code_path`"根本发不出去

我按说明书写的调用：

```jsonc
{"code_path": "…/net_v2.py", "dataset_id": "ds-001-…", "dry_run": true}
```

**回执**（不是软件回的，是**调用参数校验**被拒）：

```
1 validation error for run_customArguments
code
  Field required [type=missing, …]
```

⇒ `code_path` 是一个"**存在但无法单独使用**"的参数。

### 2.2 执行层：同时给 `code` 与 `code_path` 时，**`code` 覆盖 `code_path`**

为了绕过 2.1，我给了一个 3 行的占位 `code` 配上真的 `code_path`。回执：

```jsonc
{"ok": false, "dry_run": true, "n_params": 2,
 "error": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5 and 1x1)",
 "error_kind": "shape_mismatch",
 "error_hint": "线性层的输入特征数与权重对不上。核对 build 里的维度来源…",
 "code_path": "D:\\desktop\\量子机器学习\\施工\\nqs_docs\\analysis\\net_v2.py",
 "code_lines": 3}          // ⚠ 3 = 占位符的行数，不是文件里那 156 行
```

**两个证据**：`n_params: 2`（占位网络只有 1×1 权重 + 偏置）、`code_lines: 3`
（文件实际 156 行）。⇒ **实跑的是内联的占位符，文件压根没读。**
（但 `code_path` 字段**原样回显**了，看起来像"读到了"——这反而更容易误导。）

**修法建议（二选一）**：
- **A**：`code` 与 `code_path` **二选一必填**（校验层联动），并**禁止同时给**
  （同时给就报 `bad_code_source`，而不是静默取一个）；
- **B**：保留 `code` 必填，但允许它为空字符串以表示"用 `code_path`"。

**无论哪种，都请让回执明确说清"这次实际用的是哪一份源码"**——
现在 `code_path` 回显 + `code_lines` 是另一个来源的行数，这个组合会骗人。

### 2.3 顺带一条：`dry_run` 的回执很大

`shapes` 是**逐模块 × 每个批量**的全量轨迹。我这份网络 65 个模块 × 3 档批量
（外加保真度那一路的第二遍前向）⇒ 回执里 `shapes` 与 `runs[].shapes` **各来一份**，
同一批数据重复了两遍。**建议**：
- 顶层 `shapes` 只保留**失败那一档**或**最后一档**，`runs[]` 里保留逐档；
- 或加一个 `shapes_verbosity`（`trace` / `summary`）。

这一条不紧急，但会让 `dry_run` 在"形状确实很长"的网络（Transformer 栈）上好用很多。

---

## 3. 另外两条（不是缺口，是实测到的现状，供你判断）

### 3.1 `eval_custom` 的"自动继承训练负载"对我这个 checkpoint 没生效

回执（最新版文案）：

```jsonc
{"n_payload_arrays": 0, "payload_source": "",
 "warnings": ["这次评估**没有负载**…⚠ 正常情况下会**自动继承训练时的负载**；
               这次没有，多半是**那次训练本身没给负载**，或声明里的 HDF5 已经读不到。"]}
```

可那个运行（`custom_network-9456768-11`）**确实是带 `payload`（HDF5）训的**。
所以要么是"声明没随 checkpoint 落盘"，要么是"重读 HDF5 失败"。
**文案已经做得很好了**（把两种可能都说了），但**建议在回执里补一个原因字段**
（`payload_source_reason`：`no_declaration` / `hdf5_unreadable` / `not_persisted`），
否则 AI 只能靠猜——我这次就是靠这条警告才没有误判成"功能没做"。

### 3.2 `build(…, ctx=…)` 确实拿到了负载形状（看起来是好的）

`dry_run` 用 `n_params: 63068` 跑通了，而我的 `build` 是**从 `ctx["payload_shapes"]`**
取 `d_node`/`L` 的（**没有写死维度**）。⇒ 这条新契约在 dry_run 路径上生效。
正式训练里是否同样生效，等这轮 `job-1` 的结果落盘后再确认。

---

## 4. 一句话总结

| 新功能 | 实测 |
|---|---|
| `dry_run` | ✅ **完全达到目的**，1 个往返替代上一轮约 15 个 |
| 步级 `on_step` 指标 | ✅（已提交 `job-1`，稍后确认 tag 是否落进 TB） |
| `metric` 包 `eval+no_grad` + `metrics_every` | ✅（已在代码里用，无异常） |
| `payload_shapes` 进 `build` | ✅（dry_run 路径验证） |
| **`code_path`** | ❌ **用不了**：校验层要 `code` 必填；同时给又被 `code` 覆盖（§2） |
