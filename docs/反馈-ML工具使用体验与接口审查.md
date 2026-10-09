# 反馈：ML 工具使用体验与接口审查（开发者视角）

> **谁写的**：使用 ML Toolbox 训练神经量子态（NQS）的 AI 操作者，在**一轮 5 小时级长训练**之后，
> 把自己的遭遇交给软件开发者这一侧。
> **写给谁**：ML Toolbox 的开发者。
> **怎么来的**：**不是照抄使用者的叙述**，而是逐条到源码与现场核实——
> 引具体行号、跑只读探针量数字、在**权威进程**里做实验、扫 `runs/` 现场。
> ⚠ 所以本文有几条**推翻或修正**了使用者的自述（见 §3 对照表）：症状多半是真的，
> 但**根因常有另一个**，而"修错根因"比不修更坏。
>
> **核实结论分三档**：✅ 证实 ｜ ⚠️ 部分成立（诉求成立、理由要修正）｜ ❌ 推翻（现象另有其因）。
>
> 定位区分：使用侧（怎么用好它）见 `docs/usage-best-practices.md`；
> 接口契约见 `docs/契约.md`；踩坑清单见 `docs/pitfalls.md`。
> 核实时间：2026-10-07，现场进程 = `app_entry.py --authority`（PID 35720），
> 在跑的 run = `custom_network-2334125-0`（job-3）。

---

## 0. 结论速览

| # | 条目 | 判定 | 一句话 | 别处是否已登记 |
|---|---|---|---|---|
| A1 | 长训练中途**没有任何自动存档** | ✅ 证实 | `_save_checkpoints` 只在结果组装时调用；`trainer.train` 循环里**一行落盘都没有**。现场：`runs/custom_network-2334125-0/` 下**只有 `tb/`**（15 MB 事件文件），`checkpoint/` 目录不存在 | `SKILL.md` §中途快照、`usage-best-practices.md` §5.1（**写成使用侧绕行**，不是修好） |
| A2 | `checkpoint_detail` 答不出"这份存档是不是最新的" | ⚠️ 部分 | 它**已经报** `best_epoch` / `epochs_run`（实测回执）；真正缺的是 ①`last.pt` **落盘于第几步** ②"**用哪个顶层键 load**" ③每个 tag 各自的元信息（现在只取第一个存在的文件） | 无 |
| A3 | **没有 resume 任何入口** | ✅ 证实 | `NETWORK_HOOKS` 只有 7 个钩子（`build/loss/configure/forward_step/forward_graph/on_step/metrics`），**无加载存档的钩子**；全仓 `grep -i resume` 只命中 `api.py` 的一句英文模块说明 | `SKILL.md` §续训（使用侧绕行） |
| A4 | `run_id` 确定性 ⇒ TB 追加、曲线污染 | ⚠️ 部分 + **新增一条更硬的** | **前提已过时**：`_run_id` 早已带毫秒（`core/runner.py:52-53`），同 seed 重跑**得到不同 run_id**（实测）。你撞到的 4698 vs 102 **不是两条训练混在一起**，而是**同一个 run 里两族 tag 的 step 轴不同**（epoch 序号 vs 全局 batch 步）——**且回执里没有任何一处标注轴的语义** | `pitfalls.md` A6（记的是旧行为，并明确写"这是有意行为"） |
| A5 | 训练进程的 stdout **拿不到** | ✅ 证实 | 框架的 `Job.to_dict()`（`mecha/providers/jobs.py:83-99`）**没有任何日志字段**；`ml_mecha/` 全仓无 stdout 捕获；`read_job` 回执键里没有 `logs_tail`。job 是**权威进程里的一条线程**（`assembly.py:406`）⇒ `print` 只落在权威的控制台上 | `usage-best-practices.md` §7（写成边界） |
| A6-① | `describe_training` 默认回执过大 | ✅ 证实 | 实测 job-3 默认回执 **34 919 字符**（≈1 万 token），其中 **`curve` 占 30 557（87%）**；而 `pick` **最多只能省 ~9%**（`curve` 是**必填键**）⇒ 这是 `pitfalls.md` A33 /《反馈-观测面默认全量》缺口B 在**新字段上复发**，规模 ×10 | A33 + 缺口B（当时的修法只落在 `latest.grad_by_layer`） |
| A6-② | 自定义 tag 缺同一套判读 | ✅ 证实 | `curve.direction('sign/agree')='unknown'`、`('ph/std')='unknown'` ⇒ 即使趋势明确在爬，`convergence_note` 也回空。⚠ **"没方向的量不给诊断"这条决定本身是对的**（`curve.py:178-182` 有定论）⇒ 修法不是加启发式，而是**让网络声明方向** | 无 |
| A7-① | 存档顶层键坑 | ⚠️ 部分 | `load_state_dict(ck)`（**默认 strict=True**）**会报错**（实测 `RuntimeError` + 121 个 missing key）。**静默的只有 `strict=False`**（missing=121 / unexpected=16，无异常）——而 `SKILL.md:118` 推荐的片段**正是 `strict=False`** ⇒ 坑是按文档写出来的 | `SKILL.md` §续训坑 1、`usage-best-practices.md` §5.2 |
| A7-② | GBK 控制台 ⇒ `print('⚠')` 崩掉后台任务 | ❌ 推翻（**当前现场**）+ ✅ 潜在风险成立 | **实测无法复现**：权威进程里 `sys.stdout.encoding=utf-8 / errors=replace`、`stderr=utf-8/backslashreplace`，在权威进程里 `print('⚠')` **成功**（`ok:true`）。但这条保证是**偶然而非强制**（详见 §2.7），换一条启动路径就崩——我自己的探针脚本当场死于同一个 `UnicodeEncodeError` | `pitfalls.md` A42、`launcher.py:103-121`（都只处理**自己**的打印） |
| A7-③ | 相对路径被解析到错位置 | ⚠️ 部分 | **失败时**框架**已经**打印解析后的绝对路径（`engine.py:2422`）；真正缺的是**成功路径不回解析后的绝对路径**（`code_path` 回的是**原样入参**）⇒ "解析到另一个存在的文件"时零留痕 | `SKILL.md` 自检 #6、`usage-best-practices.md` §2.6（都只给纪律，没有机制） |

**如果只做三件**：A1（中途存档，本轮已在做）→ A4 的**轴语义**（它让人把"不同步的数"读成"矛盾"，属于最坏的假事实）→ A5（`logs_tail`，最便宜的一次往返节省）。

### 0.1 本文的处置状态（截至 2026-10-07 本轮开发）

| 条目 | 状态 | 落在哪 |
|---|---|---|
| **A1 中途存档** | ✅ **已实现** | `run_custom`/`submit_run` 新增 **`checkpoint_every`**（`param_schema` + 命令参数面；单位=**优化步**，图任务=epoch）；`trainer.train`/`train_graph` 每 N 步回调**原子**落盘 `last.pt`；`best.pt` 语义不变；失败**留痕**进 `nn_checkpoint_errors` |
| **A2 存档"落在第几步"** | ✅ **已实现** | 存档 meta 增 `saved_at_step/saved_at_epoch/saved_at/save_reason/checkpoint_every`；`checkpoint_detail` 增 **`files_meta`**（每 tag **各自**的元信息）+ **`last_saved`** + **`load_hint`** |
| **A4.2 步轴单位**（已登记为 pitfalls **A55**） | ✅ **已实现** | `describe_training` 增 **`step_units`**（只列非 epoch 级）+ **`axes`**（`batch_tags`/`mixed_tags`/`epoch_axis_len`/**`per_epoch_estimate`**）；跨族 `tags=[…]` 增 **`compare.axis_warning`** |
| §4-① `run_checkpoint`（不存在的工具） | ✅ **已修** | `checkpoint/README.txt` 改为只提真实存在的 `checkpoint_detail`/`eval_custom`，并写明"必须取 `ck['state_dict']`" |
| A7-① load 键 | 🟡 **一半**：回执已给 `load_hint`；两份文档的推荐片段改成 `strict=True` + 断言 missing | 剩下的是"用户已写下的旧代码"，只能靠文档 |
| A3 resume 入口 | ⬜ 未做 | 建议照 §2.A3 的 (a) 显式 `resume_from`（并在回执记 `resumed_from`） |
| **A5 `logs_tail`** | ✅ **已实现** | 新增 `ml_mecha/job_logs.py`：进程内装**一个 tee**，按 `threading.get_ident()` 分桶（**不能**用 `redirect_stdout`——它是进程级的，job 是线程 ⇒ 并发必串台）；桶挂 `ExecutionContext` sidecar，`read_job` 增 **`logs_tail`/`logs_n`/`logs_cap`/`logs_truncated`**（默认 10 行，`logs=0` 关）。**桶有界**（`deque(maxlen=200)`，`MLTB_JOB_LOG_LINES` 可覆盖；`logs_n` 是**真总行数**）+ **透传保留** + **编码兜底**（`⚠` 不会把日志设施或 job 搞崩） |
| **A6-① `curve` 详细度旋钮** | ✅ **已实现** | `describe_training(..., curve="brief"\|"all"\|"off")` + `curve_tags=[…]`；默认 `brief` = 核心 tag + **有判读的 tag**（+ `best_by` 点名的那个）。实测真 run（58 tag）：**38 891 → 10 664 字符（−72.6%）**，而 `convergence` 判读**一条不少** |
| **A7-② 强制 `PYTHONIOENCODING`** | ✅ **已实现** | `launcher._spawn` 给**所有子进程**注入 `PYTHONIOENCODING=utf-8`（`setdefault` ⇒ 调用方显式值优先）；`app_entry.main()` / `launcher.main()` 的 reconfigure **stdout 与 stderr 都配**（此前只配 stdout）。判据：注入断言 + **端到端**（管道里的 `print("⚠")` 必须 exit 0，并带对照组） |
| **A7-③ 回解析后的绝对路径** | ✅ **已实现** | `run_custom`（**真跑与试跑**）回执增 **`code_path_resolved`**、**`payload_resolved`**（HDF5 解析后的绝对路径 + 数据集名）与 **`code_lines`**（"跑的是哪一版"的指纹）；`_read_code_file` 的失败消息也带上解析后的位置 |
| A4 的 `run_id` 撞车检测 | ✅ **已实现**（Q5-①） | `runner.reserve_run_id()`：**自动生成**的 id 撞上已有内容（`record.json`/`tb/`/`checkpoint`）⇒ 改用 `-b<k>` 并**留痕**（`extras["run_id_collision"]` + 一行日志）；引擎侧 `_attach_progress` 同样做（它在跑之前就定 id）。⚠ **显式给的 id 一律不动**（GUI 就地重跑/固定 id 判据要的就是追加）。判据 `test_ai_efficiency.py::test_run_id_collision_is_detected_and_disentangled` |
| **Q4b 周期 best.pt**（新增） | ✅ **已实现** | 周期回调现在也收到**最佳快照**（`best_state`/`best_epoch`/`best_value`/`best_step`），最佳轮刷新后在下一个周期**原子写 `best.pt`**（`save_reason="periodic"` + `best_by` 口径）；写的是**快照而不是当前 model**。判据 `tests/test_nqs_path.py::test_periodic_best_pt_is_written_mid_run_so_an_interrupted_run_keeps_it`（训练还在跑时观测到 periodic best、序列先早轮后峰值轮、终态口径一致、权重与 last 不同） |
| **Q5-② 目标类型判定依据** | ✅ **已实现** | `prepare_dataset` 回执增 **`target_kind_reason`**（"数值且基数 18 ≤ 20 ⇒ classification"）；并把管道分层抽样那句**离根因很远**的 sklearn 错（`least populated class…`）翻成 `stratify_on_low_cardinality_target`（hint 里带基数/阈值/两个修法）。判据 `test_ai_efficiency.py::test_prepare_dataset_explains_the_target_kind_decision` |
| **Q5-③ smoke skip** | ✅ **已实现** | `tests/smoke_test.py` 新增 `_NEEDS_CALLER_CODE=("custom_network",)`，skip 时**打印理由**（"需要调用方给代码"）并注明"判据在 test_nqs_path/test_ai_efficiency，不代表被修好"；`smoke_test.py` 现在 **exit 0**（reg/cls 各 0 失败） |
| **Q3.5 fd 层盲区**（新增） | ✅ **已文档化 + 加判据**（**未**做 native 桶，理由见下） | `read_job` 回 `started_at`/`finished_at`（拿它去权威控制台按时间对齐）；空日志的 `logs_note` **点名**"C 层/子进程写 fd 1/2 的不在这里、空≠没报错"；判据 `test_native_fd_output_is_outside_logs_tail_and_note_says_so`（`os.write(2,…)` 证明边界）。**不做 native 桶的理由**：`os.dup2` 是**进程级** ⇒ 管道必须一直被排空，否则写满后**整个权威阻塞**（6 小时训练挂死在 write 上）；且 fd 1/2 被接走后原生报错**连控制台都不再出现**；它也无法归属到 job ⇒ 只能另开 `logs_native_tail` 并标注"进程级、无法归属"（`pitfalls.md` A56 写明了这三条约束，将来要做先反驳它） |
| **Q4 `best.pt` 口径可声明**（新增） | ✅ **已实现** | `Network.best_metric = "fid/sub1"` 或 `("ph/std","lower")`；值取自 `metrics`（该轮没报就跳过）；`best.pt` 按它选，`checkpoint_detail` 的 `files_meta.best.best_by` 报 tag/方向/值/轮次/步号 + `val_loss_best` 对照；**未声明逐位不变**；方向判不出来 ⇒ `bad_best_metric` 可教学地拒；整场没上报 ⇒ 退回 val_loss 并留痕。判据 `tests/test_nqs_path.py::test_best_pt_selection_is_declarable_and_default_is_unchanged` |

> 本轮新增的判据（都在 `tests/test_nqs_path.py`，随该套件进 `tests/run_all.py` 的「图/范式」阶段）：
> `test_checkpoint_is_written_periodically_and_reports_its_step`、
> `test_periodic_checkpoint_save_is_atomic_and_cleans_up_on_failure`、
> `test_step_axis_units_are_reported_and_cross_family_alignment_warns`。

---

## 1. 先说好的（这一节请**保留**，别在重构里弄丢）

| 设施 | 为什么它真的省事（实测） |
|---|---|
| **只读观测路由** `monitor_http.py`：`/latest`（最新一轮）、`/series`（整条曲线）、`/summary`、`/runs` | 训练**中**就能读（事件文件逐轮 flush）；AI 与人在看**同一份**数据，不会各说各话 |
| **`describe_training` 的 `curve`** | 起点/末值/**末段斜率**/单调性/平台占比/趋势 —— 六个数回答"这条线在往哪走"，比一个终值便宜且诚实 |
| **`convergence` 把三种病分开** | `underfit`（欠训练）/ `diverging`（发散）/ `noisy_tail`（只是抖动）。实测这条把一个"差点被我判成发散"的配置救了回来 |
| **`latest_steps`**（每 tag 各自的求值步） | 已经解决了"各 tag 求值步长不同却并排读"的一半问题（另一半见 A4） |
| **`duplicate_steps`**（同一步被写多次时如实报出） | **不静默去重**：目录脏了这件事本身是要让人看见的（这条纪律很对） |
| **`read_job(wait="terminal")` + `python -m ml_mecha.watch`** | 长训练终于不用人肉轮询；"等到超时**不是错误**"这个措辞避免了一整类误判 |
| **`run_custom` 的 `code_path` / `dry_run` / `warmup_steps`** | 调参不必重贴代码；形状类错误一次看清。这两条是这一轮最省往返的两个能力 |
| **checkpoint 的完整性**（含 bias/LayerNorm，121 张量 / 179 045 参数） | `nn_weights` 只收 `dim()>=2`，装不回来；另存完整 `state_dict` 是对的做法 |
| **`eval_custom`**（把模型交给调用方代码） | "训练完要采样/算能量/换 N 外推"这三件事**只有它能做**；`ctx["payload"]` 给张量且保住 dtype 这条约定很干净 |

---

## 2. 逐条审查

### A1 ✅ 长训练中途没有任何自动存档（最痛，且是"该由框架负责"的那一类）

**现场证据（可复现）**

```powershell
Get-ChildItem 'runs\custom_network-2334125-0' -Recurse | Select-Object FullName,Length
# 输出只有一行：
#   runs\custom_network-2334125-0\tb\events.out.tfevents.1791342346.LAPTOP-Q0CUM687.35720.2   15170351
# ⇒ 该 run 已跑到 284 epoch（loss 284 点），tb/ 15 MB，**没有 checkpoint/ 目录**
```

```
python _probe_tb.py custom_network-2334125-0     # 只读探针，见 §5 附录
#   loss      n=284  latest_step=283
#   val_loss  n=284  latest_step=283
#   fid/step  n=500  latest_step=78410
```

**根因（源码，两处，缺一不可）**

1. `ml_toolbox/methods/neural/custom.py:387`（表格路径）与 `:566`（图路径）的
   `_save_checkpoints(...)` **只在训练返回之后、结果组装时**被调用一次；
2. `ml_toolbox/methods/neural/trainer.py` 的训练循环里**没有任何落盘动作**：
   `last_state` 是循环**结束后**才构造的（`trainer.py:619`），
   也就是说——**训练期间"当前权重"根本没被物化成任何可写的东西**。

⇒ 这不是"忘了调用"，而是**能力缺失**：循环里既没有钩子，也没有参数。

**修法**（本轮已在做，见 §4 交付）

- 新增运行级参数 `checkpoint_every=N`（`param_schema` 声明 ⇒ `overrides` 与
  顶层参数两条路都能给，且非法值走 `ParamValueError` 的**可教学**失败）；
- 循环里每 N 步落一份 **`last.pt`**（**原子写**：先写临时文件再 `os.replace`）；
- **`best.pt` 语义不动**（仍按验证损失更新，仍在终态与 `last` 一起写）；
- meta 里加 `saved_at_step` / `saved_at_epoch` / `saved_at`（ISO 时间）/ `save_reason`；
- 落盘失败**不拖垮训练**（与 TB 同纪律：`tb_log` 的三处 fail-safe 是现成的榜样），
  但要**留痕**（进回执与存档，不静默）。

**判据**（已加，见 `tests/test_nqs_path.py`）

```python
def test_checkpoint_is_written_while_training_is_still_running():
    # ① 训练**还在跑时**盘上就有 last.pt：由网络自己的 on_step 在早期步读盘并上报
    # ② checkpoint_detail 报出"最后一次落盘在第几步"，且 < 总步数（证明是中途写的）
    # ③ best.pt 仍按 val_loss 选（语义未变）④ checkpoint/ 里没有残留 .tmp-*（原子性）
```

**顺带提一条设计建议（本轮没做，避免改默认行为）**：长训练**默认**开（或至少
在"预计步数很大且 `checkpoint_every=0`"时于回执里给一句 `warn`）。
理由：这类参数的失败模式是"忘了给"，而"忘了给"的代价是**5 小时**。

---

### A2 ⚠️ `checkpoint_detail` 答不出"这份存档是不是最新的"（诉求成立，理由要修正）

**现场证据（真实回执）**

```jsonc
// checkpoint_detail(run_id="custom_network-1202787-0")
{"available": true, "tags": ["best","last"],
 "files": ["README.txt","best.pt","last.pt"],
 "params": {"n_tensors": 121, "n_params": 179045, "has_bias": ["amp.bias", ...]},
 "meta": {"target_kind":"custom","n_in":1,"n_out":0,"trained_on":"cuda:0",
          "tensors_on":"cpu","best_epoch":18,"epochs_run":20,"source_lines":343}}
```

⇒ **修正**：它**确实报了** `best_epoch` 与 `epochs_run`。所以"不报第几步"的说法要收窄成三条：

1. **缺 `saved_at_step` / `saved_at_epoch`**：一旦有了 A1 的周期落盘，`last.pt` 的
   "落盘在第几步"就是**判断对账的唯一依据**（现在 `epochs_run` 只描述"跑到过第几轮"，
   不描述"这个文件是什么时候写的"）；
2. **缺"用哪个键 load"**：见 A7-①，这一条是**静默错误的直接来源**，
   而回执里明明有 `params.n_tensors` 这种"结构摘要"的位置；
3. ⭐ **只取第一个存在的文件**（`custom.py:781-795`：`if params is None:` 才赋值）
   ⇒ `params`/`meta` 全部来自 `best.pt`（循环顺序 `("best","last")`）。
   当 `last.pt` 的 `epochs_run` 比 `best.pt` 大时（周期落盘之后**必然**如此），
   回执把 `best.pt` 的数字当成**整个 checkpoint** 的数字报出去——**"假位置"那一类**。

**修法**（本轮已做）

- 每个 tag 各自的元信息：`files_meta: {tag: {saved_at_step, saved_at_epoch, saved_at, save_reason, epochs_run, best_epoch, n_tensors, n_params, mtime}}`；
- 顶层 `last_saved: {tag, saved_at_step, saved_at_epoch, saved_at}` = 按 mtime 最新的那一份；
- `load_hint: {state_dict_key: "state_dict", how: "model.load_state_dict(ck['state_dict'])"}`；
- 旧的 `params` / `meta` **原样保留**（best 优先）⇒ 既有消费者与判据零改动（加法，不改写）。

**判据**：`test_checkpoint_detail_reports_step_per_tag_and_names_the_load_key`
（断言 `files_meta.last.saved_at_step < files_meta.best.saved_at_step` 这种"两个文件不同"的情形
能被分别报出，且 `load_hint.state_dict_key == "state_dict"`）。

---

### A3 ✅ 没有 resume（续训）任何入口

**源码证据**

```python
# ml_toolbox/methods/neural/network.py:85
NETWORK_HOOKS = ("build", "loss", "configure", "forward_step", "forward_graph",
                 "on_step", "metrics")     # ← 没有 load_state / resume
```

```
grep -rn "resume" --include=*.py .    # 全仓唯一命中：
# ml_toolbox/api.py:10: ... or promise exact optimization resume.  （英文模块说明，与训练无关）
```

`run_custom` / `submit_run` 的命令参数面里也没有 `resume_from`（`ml_mecha/commands.py:222-279`）。

⇒ 使用者只能在 `build()` 里自己 `torch.load` + `load_state_dict`——
**而那条路正好踩进 A7-① 的静默坑**（SKILL 推荐片段用了 `strict=False`）。

**修法（建议，本轮未做）**：二选一，别停在中间

- **(a) 显式入口**（推荐）：`run_custom(..., resume_from=<run_id|path>, resume_which="last")`
  ⇒ 引擎装载 state_dict 后交给 `Network`（先 `build` 再 `load_state_dict`，
  并且**默认 `strict=True`**，不匹配就**响亮失败**：键集差异原样回执）；
- **(b) 钩子**：`Network.load_state(self, state) -> None`，由适配器在 `build` 之后调用。

两条都要把"**续训事实**"写进回执与 checkpoint（`resumed_from` / `resumed_at_step`），
否则"这一轮到底是从头训的还是接着训的"在审计里查不到——那是 A44 那条纪律
（身份/来源要能反查）。

**判据（比"看 loss"更硬的两条）**

1. ⭐ **参数字节级**：`resume_from=R` 之后，**第 0 步之前的 `state_dict` 与
   `R/checkpoint/last.pt['state_dict']` 逐张量 `allclose`** —— 这条**不依赖训练动态**，
   一定可复现；
2. 使用者给的"起始 loss 接近断点"作为**旁证**保留（实测 99.68 vs 1017.6，一眼可辨），
   但它会被 lr/数据顺序影响，**不该单独当判据**。

---

### A4 ⚠️ `run_id` 与"曲线污染"：前提已过时，但**真问题更硬**

#### A4.1 前提核实：`run_id` **不再是**"方法名+种子"

```python
# ml_toolbox/core/runner.py:52-53
def _run_id(method: str, seed: int) -> str:
    return f"{method}-{int(time.time() * 1000) % 10_000_000}-{seed}"
```

现场扫 2 718 个带 `tb/` 的 run：同 seed 的第二次训练得到的是**新 id**，例如
`custom_network-5595802-2002` 与 `custom_network-6122819-2002`（同 seed=2002，不同毫秒）。
`pitfalls.md` A6 记的 `method-seed` 是**旧实现**，且那条自己写着"这是有意行为，别当 bug 修"。

⇒ **"同 seed 续训会复用同一个 `tb/`"不成立**。（`SKILL.md:134-137` 与
`usage-best-practices.md` §5.3 的这条叙述**与当前实现不符**，建议按 usage-best-practices
§8.4 自己的纪律"文档与实态不符就改文档"修掉。）

**仍然成立的两条**：

- 毫秒字段是 `% 10_000_000` ⇒ **2.78 小时一个周期**：同方法、同 seed、且毫秒残差相同
  就会**撞进同一个 `runs/<id>/`**（概率极小但非零，且**撞了没有任何检测**）；
- **固定 `run_id` 的探针/扫参**会真的追加。现场抓到一例：
  `runs/linear_regression-155284-42/`（`record.json` 里 `method=ridge` 而
  `config.extras.run_id=linear_regression-155284-42`，`tb/` 下**两个**事件文件，同 PID 相隔 4 秒）
  —— 一次扫参复用了一个 `extras["run_id"]`，两次训练就叠在同一根轴上了。

#### A4.2 ⭐ 真正让人读错的东西：**两族 tag 的 step 轴不同，且没有任何标注**

```python
# ml_toolbox/methods/neural/tb_log.py
def log_epoch(...):   step = int(ep)      # ← epoch 级：轴 = 第几轮
def log_step(...):    ... add_scalar(tag, v, int(step))   # ← 步级：轴 = **全局 batch 步**
```

现场（job-3 同一份回执）：

| tag | 点数 | `latest_step` | 轴的真实含义 |
|---|---|---|---|
| `loss` / `val_loss` / `lr` / `grad/*` | 284 | **283** | 第几 **epoch** |
| `fid/step` / `dist/step` / `coef/step` / `sign/agree`（`on_step` 上报） | 500（被 `MAX_POINTS=500` 截断） | **78410** | 第几个 **batch 步**（≈284 epoch × 276 batch） |

**你给的 4698 / 102 与这个模式完全吻合**（102 epoch × 46 batch ≈ 4 692）——
最可能是把 `fid/step` 的 **`latest_step` 读成了"累计点数"**。
`read_run` 每个 tag 最多回 500 点（`tb_series.py:30`），所以**单个 tag 的点数不可能是 4 698**。
⇒ 「事件 append、两条训练混在一根轴上」这个**根因判断不成立**，但**症状是真的**，
而且比自述更值得修：

**(i) 同轴对齐会给出误导性结果（实测）**

```python
MLEngine._query_training_detail(None, run_id="custom_network-2334125-0",
                                tags=["loss", "val_loss", "fid/step"], limit=300)
# aligned_at_step = None ; aligned = {}
# 轴 = 并集尾部 300 个 = [75994 … 78410]（**全是 batch 步**）
#   loss     : [None, None, None, None, None, None]
#   val_loss : [None, None, None, None, None, None]
#   fid/step : [0.873, 0.935, 0.883, 0.928, 0.884, 0.889]
```

⇒ 回执看起来像"**这段区间根本没有 loss**"。这正好是 `pitfalls.md` A54 的病
（"把不同步的值并排放，等于制造一个假矛盾"）在 **step 尺度差 276 倍**时的极端形态：
轴被细粒度 tag 独占，粗粒度 tag 全 `null`。

**(ii) 每 tag 的"最近 500 点"覆盖的时间跨度差 276 倍，且没有一处说明**

`loss` 的 500 点 ≈ 最近 **500 epoch**；`fid/step` 的 500 点 ≈ 最近 **1.8 个 epoch**。
两个人拿各自的"最近 500 点"对比，比的是**完全不同的时间窗**。

**修法（建议）**

1. **给每条 tag 标轴的语义**：`axes: {"fid/step": {"unit": "batch", "per_epoch": 276}, "loss": {"unit": "epoch"}}`
   —— 判断依据是现成的：`on_step` 那条路（`tb_log.log_step`）就知道自己在写 batch 步；
2. **对齐时先判轴**：两族 step 的尺度比 > K（如 10×）就**明确拒绝并说明理由**，
   而不是给一张全 `null` 的表；`compare.tags` 的回执里加
   `axis_mismatch: {tags: [...], reason: "batch 步 vs epoch 步"}`；
3. **点上限改为"按轴分桶"或直接标注**：`series_limit` 旁边写清"该 tag 覆盖的 step 区间"，
   让"最近 500 点"不再是歧义句；
4. `run_id` 撞车仍建议**检测**：`runs/<id>/` 已存在且已有事件文件 ⇒ 记 `session_seq`/`started_at`
   进回执，或直接派生新 `run_id` 并记 `resumed_from`。这是"审计型工具最不该有的歧义"。

**判据**：`test_tag_axes_are_labelled_and_misaligned_alignment_is_refused`
（造一个同时有 epoch 级与 step 级 tag 的 run ⇒ 断言 `axes` 里两者 unit 不同、
且 `tags=[…]` 对齐给出**明确拒绝**而不是全 `null` 的表）。

---

### A5 ✅ 训练进程的 stdout 拿不到（这条最便宜，也最该做）

**源码证据**

```python
# D:\code-nosync\mecha\mecha\providers\jobs.py:83-99
def to_dict(self) -> dict:
    out = {"id": self.id, "state": self.state.value,
           "cancel_supported": self.cancel_supported,
           "progress": self.context.progress()}
    ...   # ← 没有 logs / stdout / output 任何字段
```

- job 是**权威进程里的一条 daemon 线程**（`ml_mecha/assembly.py:406` + `jobs.py:140`）
  ⇒ 网络代码里的 `print` 只落在**权威的控制台**上；
- `ml_mecha/` 全仓无 stdout 捕获（`grep stdout` 只命中 `dashboard.py:102` 的
  `subprocess.DEVNULL` 与 `panel_web.py` 的 reconfigure）；
- `read_job` 的回执键清单（`tools.py:1785-1796`）里没有 `logs_tail`，工具面也没有 `job_logs`。

⇒ 使用者写在 `on_step` 里的 `print("[snap] step=…")`（**证明快照真的生效了**）
对 AI 完全不可见，只能靠指标反推。

**修法（可行，机制现成）**

1. 在 `assembly.py` 的 `worker` 里给 `ExecutionContext` 挂一个 **tee 型 stdout 代理**：
   写入**同时**送到真正的 `sys.stdout`（**不能把 print 吞掉**——控制台是给人看的）
   与一个**有界环形缓冲**（如最近 200 行）；
2. ⚠ **实现约束（必须写进注释，否则一定踩）**：job 是**线程**，
   而 `contextlib.redirect_stdout` 换的是**进程级** `sys.stdout` ⇒ 直接 redirect 会让
   并发 job 的输出**串台**。正解是"**进程内装一次代理 + 按 `threading.get_ident()` 分桶**"；
3. 缓冲挂到 **sidecar**（与 `_PROGRESS_SIDECAR` 同一条路：`engine.py:143` / `:3123`），
   `read_job` 增 `logs_tail`（默认 20 行，`tail=N` 可调）；
4. 编码纪律沿用 A42：回执里的日志文本**确保可打印**（`ensure_ascii` 转义 / errors=replace）。

**判据**：`test_job_stdout_is_readable_and_still_reaches_the_console`
（① `logs_tail` 里能看到 `print` 的那一行；② 同一次 print **仍然**出现在真 stdout 上
—— 用 tee 的目标流打桩断言，防止"为了可读性把 print 吞掉"）。

**为什么这条性价比最高**：现在任何"把 print 当诊断"的用法都要靠
`run_id` + 指标反推，一次往返就为了确认一个布尔事实（"快照生效了吗"）。

---

### A6-① ✅ `describe_training` 默认回执过大（且是**已知坑在新字段上复发**）

**实测（`_query_training_detail` 的真实投影，job-3）**

```
全部 = 34 919 字符（≈1.0 万 token，按 3.5 字符/token 估）
  curve        30 557 字符   ← 87%
  latest_steps  1 783
  tags          1 500
  latest          654
```

第二个 run（`custom_network-1202787-0`，佐证）：全部 37 670 字符，其中
**`required` 骨架 = 34 189 字符**（`curve` 31 227）。

**关键点**：`_DESCRIBE_TRAINING_REQUIRED`（`tools.py:716-718`）里**含 `curve`**
⇒ `pick` 是"**补集**"投影（`tools.py:748-761`），必填键照给 ⇒
**`pick` 最多省 ~9%**。所以"用 pick 收敛"这条建议（`usage-best-practices.md` §6.3）
在这个字段上**基本无效**。

**这是复发**：`pitfalls.md` A33 与《反馈-观测面默认全量与AI调用预算》§3（缺口 B）
讲的就是"`pick` 省不动必填键里的大块"，当时的修法是给
`latest.grad_by_layer` 加 `grad_layers="summary"`。**同一个病，换了字段又长回来，
而且这次大 10 倍**（30 KB vs 3 KB）。

**修法（建议，最小改动）**

- 给 `curve` 一个**详细度旋钮**，与 `grad_layers` 对称：
  `curve="core"`（默认：`loss`/`val_loss` + 网络声明的 `primary_metric` + 已触发
  `convergence` 的 tag）/ `curve="all"`（现在的行为）/ `curve="off"`；
- 或者在回执里给 **`pickable_keys`**（可下钻的键路径清单，含 `curve.<tag>`），
  让调用方自己收敛——这也顺便回答"我能不能 pick 到必填键内部"这个反复出现的疑问；
- 两条都不改**默认语义之外**的东西：要给全量的人仍能拿到。

**判据**：`test_describe_training_default_curve_is_bounded`
（默认回执字符数上界 + `curve="all"` 时点数不变 ⇒ 证明"只是默认收窄，没有丢数据"）。

---

### A6-② ✅ 自定义 tag 只有趋势、没有判读（但**不该加启发式**）

**实测**

```python
curve.direction('sign/agree') -> 'unknown'      curve.direction('ph/std') -> 'unknown'
curve.direction('fid/sub1')   -> 'higher'       curve.direction('loss')   -> 'lower'
# 一条 0.10 → 0.80 的明显上升曲线：
curve.convergence_note('fid/sub1',   sh) -> {'kind': 'underfit', 'note': '⚠ … 可能欠训练 …'}
curve.convergence_note('sign/agree', sh) -> {'kind': '', 'note': ''}
```

**判断：框架现在的行为是对的**——`curve.py:178-182` 有明确定论：
"逐层梯度与范数类一律不给诊断，因为方向本身没有好坏"。
`sign/agree`（符号一致率）、`ph/std`（相位标准差）这类**自定义 tag 的方向只有作者知道**，
框架去猜就是"把噪声变成结论"（本仓最忌）。

**修法：让作者把方向作为契约交出来**（与 `primary_metric` 同一形状的声明）：

```python
class Network:
    primary_metric = "fid/mean"
    metric_direction = {"sign/agree": "higher", "ph/std": None}   # None = 明确不诊断
```

- 未声明 ⇒ **保持现状**（`unknown`，不诊断）—— 默认零行为变化；
- 声明了 `"higher"`/`"lower"` ⇒ `convergence` 用同一套文案判 `underfit`/`diverging`；
- 声明 `None` ⇒ 回执里显式写"作者声明不诊断"（**比沉默好**：沉默会被读成"工具漏了"）。

**判据**：`test_network_declared_metric_direction_enables_convergence_notes`
（声明后该 tag 出现判读、未声明的 tag 一字不变）。

---

### A7 静默失败三条（逐条核实，两条要改写法）

#### A7-① ⚠️ "直接 `load_state_dict(ck)` 静默全不匹配" —— **默认 strict 会报错**

**实测**（用真实存档 `custom_network-1202787-0/checkpoint/best.pt`，121 张量）

```
存档顶层键 = ['best_epoch','build_ctx','epochs_run','inline_payload','n_in','n_out',
             'n_train','n_val','params','payload_decl','source','state_dict','tag',
             'target_kind','tensors_on','trained_on']
load_state_dict(ck)                  -> RuntimeError: Missing key(s) in state_dict: "pe","subs",…  （**响亮**）
load_state_dict(ck, strict=False)    -> missing=121 unexpected=16   （**无异常 = 静默**）
load_state_dict(ck["state_dict"])    -> missing=0                    （正确写法）
```

⇒ **"无任何报错"只在 `strict=False` 下成立**——而 `SKILL.md:117-118` 推荐的续训片段
**正是** `net.load_state_dict(sd, strict=False)`（`sd = ck['state_dict'] if … else ck`）。
你的 **121** 这个数字与实测的 `missing=121` **完全一致**，所以现场确定是
"`strict=False` + 取错了键"的组合。

**修法（两条都要，成本极低）**

1. `checkpoint_detail` 直接给出**该用什么键 load**（`load_hint.state_dict_key = "state_dict"`
   + 一行 `how`）——这正是你提的"回执里直接给出该用什么键"；
2. `SKILL.md` / `usage-best-practices.md` 的片段改成**显式取键 + 默认 strict**，
   或者至少把 `missing_keys` 断言掉（`assert not m.load_state_dict(...).missing_keys`）。
   ⚠ **文档推荐的写法必须是"错的会响"的那种**——现在的写法是"错的会静默"。

**判据**：`test_checkpoint_detail_names_the_load_key_and_strict_load_is_clean`
（`load_hint.state_dict_key` 存在；用整份 ck 做 `strict=True` 必须抛；用 `ck["state_dict"]` 必须 0 missing）。

#### A7-② ❌ "GBK 控制台 ⇒ `print('⚠')` 让后台任务整个崩掉" —— **当前现场无法复现，但风险真实**

**实测（在权威进程里跑代码：`eval_custom`，与 job 同一进程、同一 stdout）**

```
第一次：print("[probe] ⚠ non-ascii")   ->  ok: true
第二次：读回编码事实 -> metrics = {
          "stdout_enc_utf-8": 1.0,       "stdout_err_replace": 1.0,
          "stderr_enc_utf-8": 1.0,       "stderr_err_backslashreplace": 1.0,
          "stdout_isatty_True": 1.0,     "env_pyioencoding_unset": 1.0}
```

⇒ 权威进程（PID 35720，`app_entry.py --authority`，由 `launcher.py` 以
`CREATE_NEW_CONSOLE` 起）的 **stdout 是 utf-8 + errors=replace**。
原因有两层：真控制台（Python 用 `WindowsConsoleIO`）+ `app_entry.main()` 主动
reconfigure（`app_entry.py:137-141`）。**所以"必然崩"不成立。**

**但这条保证是偶然而非强制**（四点，全部可核实）：

1. `PYTHONIOENCODING` **未设**（实测 `env_pyioencoding_unset`）；
2. `launcher._spawn(cmd, env=None)`（`launcher.py:125-127`，权威那处 `env=None`）
   ⇒ **不注入任何编码环境变量**；
3. reconfigure 只在 `main()` 被走到时发生，而且**只配 stdout**（stderr 靠控制台默认）；
4. **一旦 stdout 不是真控制台**（`> log.txt`、被上层程序按管道捕获、或以库方式在进程内跑），
   编码就退回 **cp936** ⇒ 同一个 `print('⚠')` 立刻致命。

**这条我有现场证据**：本报告的探针脚本（普通 `python` 子进程，stdout 是 pwsh 的管道）
**当场就死了**：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0' in position 50
```

**修法**（把"偶然"改成"强制"，两处）

1. `launcher._spawn` **统一注入** `env = {**os.environ, "PYTHONIOENCODING": "utf-8"}`
   （对所有子进程生效：权威、dsh、面板、watch）；
2. `app_entry.py` / `launcher.py` 的 reconfigure 把 **stderr 一起**配
   （现成榜样：`panel_web.py:1027-1032` 就是 for stream in (stdout, stderr)）；
3. `SKILL.md` 顶部"网络代码里的 print 只用 ASCII"这条纪律**保留**（它挡的是
   别人机器上的同类问题），但从"必须"降级成"保险"。

**判据**：`test_child_processes_force_utf8_io_encoding`
（① 断言 `_spawn` 的 env 里必有 `PYTHONIOENCODING=utf-8`；② 一个 `print('⚠')` 的子进程
在**管道**捕获下退出码 0 —— 放回不注入的版本立刻红）。

#### A7-③ ⚠️ 相对路径：**失败时已经报了绝对路径，成功时没报**

**源码（已有的一半）**

```python
# ml_mecha/engine.py:2420-2425
if not candidate.exists():
    raise MechaError(f"数据文件不存在：{str(candidate)!r}",     # ← 已含**解析后的绝对路径**
                     kind="dataset_not_found",
                     hint="相对路径按装配时声明的 dataset_roots[0] 解析")
```

**缺的一半**：成功路径**不回解析后的绝对路径**——
`run_custom` 回执里的 `code_path` 是**原样入参**（`engine.py:1468`），
HDF5 负载声明里也是原样路径。⇒ 当 `../../data/x.h5` **解析到了另一个存在的文件**时，
**零留痕**：回执看着"读到了"，读的是别处的数据。这与你"等了 120 分钟才发现"的
现象一致（**不是文件不存在，而是存在但不是你以为的那一个**，或者错在更下游）。

**修法**

- 回执里加**解析后的绝对路径**：`code_path_resolved`、负载的
  `payload_resolved`（HDF5 路径与数据集名）；
- `describe_dataset` 的 `ledger` 里也带上来源的绝对路径（现在只有声明）。

**判据**：`test_receipt_echoes_resolved_absolute_paths_for_code_and_payload`
（给相对路径 ⇒ 回执里的 resolved 字段必须是绝对路径，且与 `dataset_roots[0]` 拼接结果一致）。

---

## 3. 我核实/推翻的清单（对使用者自述的逐条更正）

| 自述 | 核实结果 | 更正 |
|---|---|---|
| "一个跑到 130 epoch 的 run 只有 `tb/`" | ✅ **完全成立** | 现场更狠：job-3 已 284 epoch、15 MB 事件文件，仍无 `checkpoint/` |
| "`_save_checkpoints` 在结果组装时才调用" | ✅ 成立 | 补一句更要紧的：**循环里连"当前权重"都没物化**（`trainer.py:619` 在循环之后） |
| "`checkpoint_detail` 不报第几步" | ⚠️ 需收窄 | 它报 `best_epoch`/`epochs_run`；缺的是 `saved_at_step`、**load 用哪个键**、**每个 tag 各自的 meta** |
| "没有加载存档的钩子" | ✅ 成立 | `NETWORK_HOOKS` 7 个钩子里没有 |
| "`run_id` = 方法名+毫秒+种子 ⇒ 同 seed 复用同一 `tb/`" | ❌ **前提不成立** | 毫秒字段让同 seed 得到**不同** id（现场扫 2 718 个 run 验证）；真会追加的是**固定 run_id 的探针/扫参**（抓到 `linear_regression-155284-42` 一例） |
| "`fid/step` 4 698 点而 `loss` 102 点 ⇒ 两条训练混在一根轴上" | ❌ **根因不成立** | 4 698 与 102 的比值 ≈ 每 epoch 的 batch 数；这是**两个 step 轴**（batch vs epoch）。`read_run` 单 tag 上限 500 点，**不可能**回 4 698 个点 |
| "按 `wall_time` 过滤也切不干净" | — 未复现 | 单 run 单事件文件时 `wall_time` 过滤无从下手；`duplicate_steps` 已能报"同一步被写多次" |
| "读不到后台任务的 stdout" | ✅ 成立 | 框架 `Job` 结构里根本没有日志字段 |
| "默认回执几万 token" | ✅ 成立（数量级对） | 实测 34 919 字符 ≈ 1 万 token；**且 `pick` 救不了**（`curve` 是必填键，87%） |
| "`sign/agree`、`ph/std` 只有趋势没有判读" | ✅ 成立 | 但**框架的做法是对的**（方向未知就不诊断）⇒ 修法是让作者**声明方向** |
| "直接 `load_state_dict(ck)` 静默全不匹配、无报错" | ⚠️ 需改写 | 默认 `strict=True` **会报错**；静默的是 `strict=False`——而 SKILL 推荐的片段正是它。121 这个数字与实测 `missing=121` 完全一致 |
| "`print('⚠')` 让后台任务整个崩掉" | ❌ **当前现场不可复现** | 权威 stdout 是 utf-8（实测读数）；崩的是**普通 cp936 管道子进程**（我的探针当场崩）。风险真实但成因不同 ⇒ 修法是**强制注入 `PYTHONIOENCODING`** |
| "报错时把解析后的绝对路径打出来" | ⚠️ 已做一半 | **失败时已经打了**（`engine.py:2422`）；缺的是**成功时**回解析后的路径 |

---

## 4. 顺带发现的四个小问题（都是"文档/回执与实态不符"这一类）

1. ⭐ **`checkpoint/README.txt` 让人用一个不存在的工具**：
   `custom.py:1047` 写"装载：`run_checkpoint(run_id=…)` 拿定位符"，
   而全仓 `grep run_checkpoint` **只有这一处字面量**，工具面里叫 **`checkpoint_detail`**。
   这正是 `pitfalls.md` A21（"文档推荐的用法从没被跑过"）。**修法**：README 文案改对；
   **判据**：一条静态判据——「README/SKILL 里出现的工具名必须都在工具面注册表里」
   （比人工核对可靠）。
2. **`describe_checkpoint` 把 `best.pt` 的 meta 当整个 checkpoint 的**（见 A2-3）：
   周期落盘上线后必然暴露（两个文件的 `epochs_run` 会不同）。
3. **`SKILL.md` / `usage-best-practices.md` 的 run_id 叙述与实现不符**（见 A4.1）：
   应改为"两族 tag 的 step 轴语义不同（A4）+ 固定 run_id 的扫参会追加"。
4. **`latest_steps` 有了步号但没单位**：正因为它已经存在，补一个 `unit`
   （`epoch` / `batch`）就能把 A4.2 的一半问题就地解决——**最便宜的改法**
   （✅ 本轮已实现：`step_units` + `axes`）。
5. **连续目标被"基数阈值"判成分类，而报错离根因很远**（本轮写判据时踩到）：
   `contracts.infer_kind` 的规则是"**数值且基数 > 20** 才算回归"
   （`REGRESSION_CARDINALITY = 20`）⇒ 一份 24 行（切 8:2 后 18 行）的**连续**目标
   被判成 `classification`（18 个"类别"），于是自定义网络的 `MSELoss` 报
   `RuntimeError: Found dtype Long but expected Float` —— 使用者会去查网络代码，
   而真相是"行数/基数太少"。建议：**在 `prepare_dataset` 的回执里把判定依据说出来**
   （"目标基数 18 ≤ 20 ⇒ 判为 classification；要强制回归请显式声明"），
   并在自定义网络那条路上把这个错翻译成一句可教学的话。

---

## 5. 附录：复现用的只读探针

> ⚠ 2026-10-07 起这些脚本移到 **`checks/oneoff/`**（Q6 清理：仓根不再堆一次性脚本；
> 它们**不被任何代码 import**、也不在门禁里）。见 `checks/oneoff/README.md`。

```powershell
# ① 现场：一个 run 目录里到底有什么
Get-ChildItem 'runs\custom_network-2334125-0' -Recurse | Select-Object FullName,Length

# ② 逐 tag 点数与 latest_step（只读，不跑训练）
python checks\oneoff\_probe_tb.py custom_network-2334125-0 custom_network-1202787-0

# ③ 同 run 内多 tag 对齐（A4.2 的误导性证据）
python checks\oneoff\_probe_align.py custom_network-2334125-0

# ④ 默认回执的体积构成（A6-①）
python checks\oneoff\_probe_receipt.py custom_network-2334125-0

# ⑤ 存档顶层键 / strict 行为 / tag 方向（A7-①、A6-②）
python checks\oneoff\_probe_ck.py custom_network-1202787-0

# ⑥ 权威进程里的编码事实（A7-②，经 eval_custom 在权威进程执行）
#    mcp: eval_custom(run_id="custom_network-1202787-0", code="def evaluate(m,ctx): ...")

# ⑦ ⭐ **长期有用的事实核查**（A56 的"不成立的理由"就是它推翻的）：
#    `stdout=PIPE` 时 `app_entry.main` 的 utf-8 reconfigure 还成立吗？
Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue   # ⚠ 必须先剥掉
python checks\oneoff\_factcheck_isatty_utf8.py
#  期望：PIPE 时 BEFORE gbk isatty=False → AFTER utf-8 isatty=False；不调 main 的对照是 gbk
```

> 探针记录的数字就是本文每一处断言的出处；想让某条读数长期有效，
> 请把它**固化成判据**（本轮的读数已固化进 `tests/test_nqs_path.py` /
> `tests/test_ai_efficiency.py` / `tests/test_ml_mecha.py` / `tests/test_ml_delivery.py`）。

---

## 6. 优先级建议

| 顺序 | 条目 | 成本 | 不做会怎样 |
|---|---|---|---|
| 1 | **A1 周期存档**（本轮已做） | 中 | 5 小时训练中途死机 ⇒ 全部参数丢失 |
| 2 | **A4.2 轴语义标注 + 错轴对齐拒绝** | 低 | 继续让人把"不同步的数"读成"矛盾"，并把"没有 loss 数据"读成事实 |
| 3 | **A5 `logs_tail`** | 低 | 每次确认一个布尔事实都要付一次往返；网络代码里的 print 形同废弃 |
| 4 | **A6-① `curve` 详细度旋钮** | 低 | 每次默认调用付 ~1 万 token；`pick` 无效会持续误导用法 |
| 5 | **A7-① load 键 + A7-② 强制 utf-8** | 极低 | 一个静默错（模型等于随机）+ 一个随机崩 |
| 6 | **A3 resume 显式入口** | 中 | 续训只能走"用户自己打补丁"的路，而那条路正是 A7-① 的坑 |
| 7 | **A2 / A7-③ / §4 的小问题** | 低 | 精度与可追溯性（审计型工具的立身之本） |

---

## 7. 附录 B：本轮实现清单与判据结果（2026-10-07）

### 7.1 改了哪些文件

| 文件 | 改了什么 |
|---|---|
| `ml_toolbox/methods/neural/trainer.py` | `train()` / `train_graph()` 新增 `checkpoint_every` + `on_checkpoint` 参数；循环里每 N 步回调（表格路径=**优化步**，图路径=**epoch**）；新增 `_try_checkpoint()`（**失败不拖垮训练、但必须留痕**）与 `DEFAULT_CHECKPOINT_EVERY=0`；`stats["checkpoint"]` 回报 `n_ok`/`last_step`/`errors` |
| `ml_toolbox/methods/neural/custom.py` | `param_schema` 增 `checkpoint_every`；新增 `_atomic_torch_save()`（临时文件 + `fsync` + `os.replace`）、`_checkpoint_common()`（两份存档共用构造）、`_make_periodic_checkpoint_saver()`、`_write_checkpoint_readme()`（**并修掉 README 里的 `run_checkpoint` 假工具名**）；`_save_checkpoints()` 增 `saved_at_step/epoch/save_reason/checkpoint_every` 且改原子写；`describe_checkpoint()` 增 **`files_meta`/`last_saved`/`load_hint`/`interrupted_saves`**（旧的 `params`/`meta` 原样保留）；两条路径都写 `nn_checkpoint_*` artifacts |
| `ml_mecha/commands.py` | `run_custom` 增一等参数 `checkpoint_every`（`CommandParam` + `extra_receipt_keys`） |
| `ml_mecha/engine.py` | `_cmd_run_custom`：把顶层 `checkpoint_every` 合流进 `overrides`（**顶层为准**，只搬运不校验）+ 回执回显；`_query_training_detail` 增 **`step_units`/`axes`/`compare.axis_warning`**；新增 `_step_unit_of()` / `_axes_facts()` / `_int_or_zero()` |
| `ml_mecha/tools.py` | `run_custom` / `submit_run` 增 `checkpoint_every` 参数与 schema；`run_custom` 回执回显它 |
| `tests/test_nqs_path.py` | 新增 3 条判据（见 7.3）+ 登记进 `_TESTS` |
| `tests/test_ml_delivery.py` | null-shim 探针同步：`EXPECTED_DEFAULTED` 的两个工具各 +`checkpoint_every`、`_DEFAULTED_PROBES` +2、`_EXPECTED_OPTIONAL_PARAMS` 74→**76**（**新增参数必须补探针**，该判据自身就是这么设计的） |
| `skills/ml-toolbox/SKILL.md` | §中途快照→**§中途存档（`checkpoint_every`）**；§续训改成 `strict=True` + 断言 missing，并**收敛成"权威版见 usage-best-practices"**（消掉两处重复叙述）；工具表/§0 自检/情形 C/结果拿回来 同步；**新增"读数值前先看轴"** |
| `docs/pitfalls.md` | **A6 更正**（标注"旧行为，已改"+ 残留风险 + 指向 A55）并**去掉重复的两份正文**；新增 **A55**（两族 step 轴单位） |
| `docs/usage-best-practices.md` | §5.1/§5.4/§0/§2/§4/§6.1/§6.2/§6.3/§7 与新能力对齐（长训练 checklist 改成"给 `checkpoint_every`"） |
| `docs/反馈-ML工具使用体验与接口审查.md` | 本文（§0.1 处置状态 + 本节） |

### 7.2 新参数（用法）

```jsonc
run_custom(..., checkpoint_every=200)                 // 每 200 个优化步原子落盘 last.pt
submit_run(code_path=…, checkpoint_every=200)         // 后台长训练同样给
run_custom(..., overrides={"checkpoint_every": 200})  // 等价；同时给时**顶层为准**
```

- **默认 0 = 关**（逐位等价于从前）；单位：表格/负载路径=**优化步**，图路径=**epoch**；
- `best.pt` 语义不变；周期只写 `last.pt`，终态再写 `best.pt`+`last.pt`（`save_reason=final`）；
- 失败**不抛**（训练照常）但留痕：`artifacts["nn_checkpoint_errors"]`、`diag.checkpoint_periodic`。

### 7.3 判据命令与结果

```powershell
python -m pytest tests/test_nqs_path.py -q
#  22 passed   （含 3 条新判据）
```

| 新判据 | 钉住什么 |
|---|---|
| `test_checkpoint_is_written_periodically_and_reports_its_step` | **训练线程还活着时**盘上就有 `last.pt`（另起线程轮询，`seen[rid] is True`）；`nn_checkpoint_n >= 2`；`files_meta` 每 tag 都有 `saved_at_step/epoch`；`last_saved.tag == "last"`；`load_hint.state_dict_key == "state_dict"`；无 `.tmp-*` 残留 |
| `test_periodic_checkpoint_save_is_atomic_and_cleans_up_on_failure` | 成功路径不留临时文件；**失败路径清临时文件且不毁上一份好存档**（不可序列化对象触发） |
| `test_step_axis_units_are_reported_and_cross_family_alignment_warns` | `step_units["my/step"]=="batch"`、epoch 级不出现；`axes.per_epoch_estimate`；跨族 `tags=[…]` 带 `axis_warning` |

**逐套件回归（串行跑，避免并行假红）**

| 套件 | 结果 |
|---|---|
| test_nqs_path / test_ml_mecha / test_ml_delivery / test_ml_capabilities | **22 / 58 / 63 / 9 全通过** |
| test_ai_efficiency / test_skill_surface / test_tensorboard / test_regressions | **17 / ✓ / 10 / 37 全通过** |
| test_problem / test_panel_web / test_plot_spec / test_ui / test_opt / test_dsh_panel | **7 / 18 / 8 / 20 / 43 / 2 全通过** |
| `smoke_test.py` | ⚠ **预存在失败**：`custom_network` 报"没有拿到网络代码"（2 处）。**与本轮无关**——`git show HEAD:ml_toolbox/methods/neural/custom.py` 里 `_register_into_registry()` 早就在 import 时执行，而 `smoke_test` 走的是"不给代码的通用方法路径"⇒ 它**必然**失败。建议把 `custom_network` 加进 smoke 的 skip 清单并写明理由（"需要调用方提供代码，通用路径无法跑"） |

> ⚠ **一条环境教训（顺带记下）**：`tests/run_all.py` 三阶段并行时，本轮曾出现 **11 套件同时红**，
> 报的全是 `WinError 1455 页面文件太小`（连纯 stdlib 的 `test_skill_surface` 都崩在 import）
> ——根因是**权威进程里还有一个 GPU 长训练在跑**，提交内存不够。
> 逐套件**串行**重跑后**全部转绿**。⇒ 这不是代码回归，是 `tests/test_nqs_path.py:25-29`
> 已经写下的那类"并行假红"；判读红色时**先看是不是 import 期就死了**。
