# 反馈：NQS 路径实测缺口（AI 侧实跑后登记）

> **谁写的**：一个用 ML Toolbox 训练神经量子态（NQS）的使用者（AI 操作）。
> **写给谁**：ML Toolbox 的开发者。
> **怎么来的**：不是读代码推断，而是**真的把一次 NQS 训练从头跑完**——
> 登记逐行负载 → 跑自定义网络（GNN + Transformer 自回归 + 复数输出头 + 相位预热）
> → 读逐 epoch 观测 → 取 checkpoint → 用 `eval_custom` 装载模型做自回归采样。
> 全链路**已跑通**（见 §1 证据）。本文只登记跑通之后仍然卡人的地方。
>
> 需求原文见 `docs/需求-在ML工具里训练神经量子态.md`。
> 好消息先说：那份需求里的十条，**九条已经能用**，而且用法比预期顺。
> 本文的四条是"能跑但会咬人"或"自己的参数组合走不通"。

---

## 1. 已经打通的部分（作为对照基线，不必再改）

实测通过的链路与证据（run `custom_network-4304705-11`）：

| 环节 | 实测证据 |
|---|---|
| 逐行负载（`arrays` 与 `hdf5` 两种） | `prepare_dataset` 回执逐数组给 `shape`/`dtype` |
| 复数保真 | `supp_amp` complex64 原样到网络；`bits`(int8) 未被 float 强转 |
| `target_kind="custom"`（无标准标签） | 不造假指标，`primary_metric` 由网络声明 |
| GPU 训练 | checkpoint 元信息 `trained_on: cuda:0` |
| 复数输出头 + 自回归 | 模长/相位两路，逐位 softmax 归一 |
| **相位预热** | `on_step` 生效：`grad/ph.weight ≈ 0.43`，其余 18 层 ≤ 0.045 |
| 自定义指标 + 分组 | `fid/sub0`、`fid/sub1`、`fidmax`/`fidmin` 全部进 TB 与回执 |
| 逐 epoch 观测（含逐层梯度） | 19 个 `grad/*` tag 一次读全 |
| checkpoint | `best.pt` / `last.pt`，51 张量 / 7454 参数 / `best_epoch=18` |
| **模型取回 + 采样 + 对照精确值** | `eval_custom` 装载 best，采 256 构型，回 `occ_err_vs_exact` |

`on_step` / `metrics` / `forward_step` 这三个钩子的设计**正中要害**：
尤其"tag 命名自带分组（`fid/ch0`）而查询面一行不用改"这一条，是很干净的设计。

---

## 2. 缺口清单（按建议修复顺序）

### 缺口 1【最高】`build()` 拿不到负载 ⇒ 输入维度只能猜

**现象**：`Network.build(self, n_in, n_out, *, nn)` 只拿得到**特征表列数**。
而 NQS 的输入维度在**负载**里（节点特征几维、每条配置几个原子、几个采样构型），
`forward_step` 才第一次看到负载。于是 `build` 里只能**写死维度**：

```python
def build(self, n_in, n_out, *, nn):
    return GNN()            # ← 内部 self.feat = nn.Linear(5, 6)，5 是猜的
```

**为什么这是最高优先**：写错的后果**不是报错，而是换一份数据（换个 N、加一维特征）
时静默跑偏**。这正好违背本仓"不静默"的一贯纪律，也是 AI 最容易犯的错
（AI 拿不到事实，就只能编一个）。

**建议**：给一个在 `build` **之前**调用的钩子，把负载形状先交出来：

```python
def init(self, ctx):        # 可选；在 build 之前调一次
    # ctx: payload_shapes={"xr": (R, L, 5), ...}, n_rows=R, n_features=..., target_kind=...
    self.d_node = ctx["payload_shapes"]["xr"][-1]
```

**位置**：`ml_toolbox/methods/neural/network.py`（契约）、
`custom.py:180-185`（`n_in = int(prep.Xs.shape[1])` 之后、`call_network_build` 之前）。

---

### 缺口 2【高】`eval_custom` 带 `dataset_id` 直接崩（确定性 bug）

**复现**：

```
eval_custom(run_id=…, which="best", dataset_id="ds-005-…", code="def evaluate(m,c): return {}")
```

**回执**：

```json
{"ok": false, "error": {"message": "AttributeError: 'dict' object has no attribute 'spec'",
 "info": {"kind": "internal_error", "hint": "栈留在宿主日志；回执只给结构化失败"}}}
```

**根因**（已定位到行）：`ml_mecha/engine.py:956-957`

```python
entry = self._validator.check_dataset_ref(ds_id)
payload = dict(getattr(entry.spec, "aux", {}).get("payload") or {})
```

而 `ml_mecha/validator.py:515` 的 `check_dataset_ref` **返回的是 facts 字典**
（`dataset_facts(...)`），不是带 `.spec` 的 DatasetEntry ⇒ `getattr(entry.spec, …)`
必然 `AttributeError`。

**影响**：技能文档 `SKILL.md` §"训练完把模型拿回来"推荐的三个工具串联里，
`eval_custom` 的 `dataset_id` 是**文档写明的用法**，但它走不通。
用户只能把负载**原样再传一遍**绕过（我实跑就是这么绕的）。

**建议**：要么让 `check_dataset_ref` 回 entry（或另取一个 `dataset_entry(ds_id)`），
要么在 engine 侧从 facts 里取负载。这条修完，"复用会话里某份数据集的负载"才名副其实。

**另一条相关的（同一处体验）**：`eval_custom` **不继承训练时的负载**。
`ctx["payload"]` 只含本次调用收到的。技能里"换一个没见过的 N 评估"的示例写法
会直接 `KeyError: 'xr'`。要么在 `ctx` 里带上训练负载的引用，要么在文档里写清
"必须显式给 `dataset_id` 或 `payload`"。

---

### 缺口 3【高】自定义任务的验证轮仍调 `model(Xva)` ⇒ 逼出一个**假 forward**

**现象**：`target_kind="custom"` 的训练跑完第一轮，进验证时报：

```
RuntimeError: Module [GNN] is missing the required "forward" function
```

**根因**：`ml_toolbox/methods/neural/trainer.py` 的验证分支
（`train()` 里 `if Xva is not None:` 那一段）**先**走 `o = model(Xva)`
——即使 `has_y is False`、即使 `forward_step` 已经给了 `(out, target)`。

**我被迫写的东西**：

```python
def forward(self, x):
    return self.feat(x).sum(-1)      # ← 与 NQS 损失毫无关系，只为让验证别炸
```

**这条为什么严重**：训练器的 `best_epoch` / `best_watch` 是**按 `val_loss` 选的**
（`trainer.py` 的 `watch = vl if vl is not None else train_loss`）。
自定义任务的 `vl` 由**这个假 forward** 算出 ⇒ **`best_epoch` 对 NQS 完全无意义**，
`keep_best` 存下来的"最佳轮"是假的。而 `checkpoint_detail` 会把这个假的
`best_epoch=18` 当事实回报给 AI —— 正是本仓最防的那种"假绿"。

**建议**：二选一，别停在中间：
- **(a) 尊重契约**：`has_y is False` 时验证损失一律走 `forward_step`
  （文档里已经这么承诺了："自定义任务：验证损失也得由网络的 `forward_step` 算"）；
- **(b)** 若确实需要 `model(x)` 兜底，则**在文档里明确要求自定义任务必须实现
  `forward`**，并说明它同时会被用于 `best_epoch` 选择——让作者知道这个返回值有语义，
  不要写成假的。

**位置**：`ml_toolbox/methods/neural/trainer.py`（`train()` 的验证段）。

---

### 缺口 4【中】负载数组的"多带一维"要求没有任何提示，且**静默塌陷**

**现象**：我把逐配置特征写成 `(R, 5)`（以为"第 0 维 = 行数"就够了），
`prepare_dataset` **收下了**，回执 `payload.xr = {shape: [6, 5], dtype: float64}` 看着完全正常。
但 `trainer.py:384` 的 `v[row_idx]` 切出来是 `(B, 5)`，而网络期待 `(B, L, 5)`
⇒ 报出来的是 `mat1 and mat2 shapes cannot be multiplied` 这种**离根因很远**的错。

**正解**（我试出来才明白）：负载数组要**多带一维**——
`(R, L, 5)` 才是"每行 L 个原子、每个原子 5 维特征"。

**建议**：在 `prepare_dataset` 的回执里，对**每个负载数组**给出"切批后形状"的提示，
或在文档 §逐行负载 里加一句：
> 数组形状是 `(R, …)`；**`…` 部分就是"一行之内"的完整结构**。
> 若一行是一个长度为 L 的向量序列，要给 `(R, L, d)`，不要给 `(R, d)`。

这条不改代码也能缓解（纯文档），但**改了能省掉每个新用户的一次踩坑**。

---

### 缺口 5【低】`metrics` 在 `model.train()` 状态下被调用，且每轮回调一次

**现象**：`trainer.py` 里 `metrics(model, _ctx(None, ep, kind="metrics"))` 在
`model.train()` 之后调用 ⇒ 没有 `no_grad`。我不得不自己写 `with torch.no_grad():`，
否则整份负载的保真度计算会建图、吃显存。

**第二条**：`metrics` 每轮回调**一次**且传全量负载。对 NQS 这种"每轮要在
全量 2^N 支撑上算保真度"的指标很贵；对 974 组配置的数据集会直接拖垮训练。
我这次只用 6 行才跑得动。

**建议**（都不急）：
- `metrics` 调用处包 `torch.no_grad()`（观测不该有梯度语义）；
- 文档里写明"`metrics` 每轮调一次、传全量负载，**别在里面对全量数据做重活**"，
  或给一个"每 N 轮算一次"的旋钮。

---

## 3. 两条实测经验（建议进 `SKILL.md` 或 `pitfalls.md`）

这两条我**加起来烧了十几次往返**，而且报出来的错都和根因无关：

### 3.1 在自定义网络里建张量，**一律用 `device=`**

```python
# ❌ 模型在 cuda，这个在 cpu ⇒ 报的是"维度不匹配"这种离根因很远的错
a = b.unsqueeze(-1) * torch.tensor([0.0, 1.0])

# ✅
a = b.unsqueeze(-1) * torch.tensor([0.0, 1.0], device=b.device)
```

我实测：`oh()` 里一个 `torch.tensor([0., 1.])` 没写 device，
报出来的是 `Sizes of tensors must match except in dimension 2` 和
`not enough values to unpack (expected 3, got 2)` —— **全都是误导**。
`ctx["device"]` 已经在 ctx 里了，但"建张量要带 device"这条纪律
只在图任务那一节写了（"建张量要用 `ctx['device']`"），**表格/负载路径没写**。
建议提到契约的通用纪律里，并补一句"设备不一致的错误信息会伪装成维度错"。

### 3.2 自回归的"序列长度"是 `L`（原子数），不是 `K`（每行采几个构型）

这条纯粹是我自己的坑，但值得写进示例：负载 `bits` 是 `(R, K, N)`，
`K` 是"每行采了多少个构型"、`N` 才是自回归的序列长度。
把 `K` 当序列长度会导致一连串形状错。文档里给一个**最小自回归示例**（含
`(R, K, N)` 的语义标注）会比文字说明省很多时间。

---

## 4. 附：我为了让链路跑通做的临时绕行（供你判断优先级）

| 缺口 | 我的绕行 | 代价 |
|---|---|---|
| 2（`eval_custom` 崩溃） | 把负载内联**再传一遍**给 `eval_custom` | 每次评估都要重传几十 KB |
| 3（假 forward） | 写一个与损失无关的假 `forward` | `best_epoch` 失真，只能事后用 `eval_custom` 自己重算 |
| 1（维度靠猜） | `build` 里写死维度 | 换数据就静默错 |
| 4（静默塌陷） | 手工实验出"要多带一维" | 十几次往返 |

**如果只修一条，我建议修缺口 3**：它让"最佳轮"变成假事实，
而这是本仓最在意的那类问题（假绿）。缺口 2 是纯 bug，改起来最快。
缺口 1 决定"换数据安不安全"，是长期收益最大的一条。
