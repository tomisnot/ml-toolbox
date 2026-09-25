# ML Toolbox 的 dsh 插件（驾驶舱监控面板）

这个目录是 **P2c** 的产物：把 EL 的 dsh 驾驶舱面板移植到 ML，并做**参数化 + 改名**。
工具接入**不**经过本插件——走 dsh 内置 `@deepseek-ai/dsh-mcp-client`（方案 A，
见 `docs/mecha/07-交付形态-双模式与dsh桥.md` §4）。

## 它做什么

| 半 | 文件 | 职责 |
|---|---|---|
| HOST | `src/index.ts` → `lib/index.mjs` | 在 dsh **同源** webserver 上注册只读路由 `/ml-toolbox/monitor-url`，返回 `{base: 'http://127.0.0.1:<监控端口>'}`（端口读自权威写的运行期描述符，**没有硬编码默认值**） |
| CLIENT | `src/client/*` → `lib/client.js` | 会话头「◈ 监控」按钮 + 右栏 tab（三页：飞行记录仪 / 配置态 / 运行记录），轮询权威的只读端点 |

**不移植** EL 的 `dsh/src/host/mcp-bridge.ts`（自愈 MCP 桥）。那条桥是为治
dsh 内置客户端的 streamable-http "HTTP 世代"死区而写的；ML 先靠"权威与 dsh
同生共死 + 每次启动重生成 overlay"绕开它。**什么时候该上它**见下面的观测程序。

## 为什么是"同源路由"，不是"一个 command"（真机教训）

第一版注册的是 `/ml-monitor-url` **command**，由 client 经
`remote.commands.execute(sessionId, …)` 取地址。真机上挂在两处：

1. **返回值形状脆弱**：command 的返回类型是
   `RemoteResult<CommandExecution | undefined>`，而
   `CommandExecution = { commandId, result: CommandResult }`、
   `CommandResult = { kind:'success', text? } | { kind:'error', text }`
   ⇒ 地址在 `r.value.result.text`。当时按 `r.value.text` 读 ⇒ **永远取不到**，
   面板显示"监控端点未知：命令未返回监控地址"。（当时那条自测把 `r.value`
   按错形状 stub 了，所以**自测绿、真机必挂**——自证假绿。）
2. **取一次地址会污染会话**：每次 `commands.execute` 都会往会话里写
   `command/run` + `command/done` 两条记录，而面板只是想知道一个端口。

改为挂在 dsh 同源 webserver 上的只读路由：
`webServer.register({kind:'exact', path:'/absolute/no/trailing/slash', handler})`
（handler 拥有完整响应生命周期，返回 disposer）。面板直接 `fetch` 同源路径：
无返回值包装、无跨源、无会话副作用。

> **判据现在喂真形状**（`{ok, status, json}`）+ 一条**负例**：把旧的 command
> 返回形状喂进去**必须失败**。这样"又把地址读成命令返回值"会被当场抓住。

## 数据根从哪来（`MLTB_ML_ROOT`）

host 半读运行期描述符时会先看环境变量 **`MLTB_ML_ROOT`**，没有才用
`process.cwd()`。看门人起 dsh 时显式设置它。

**为什么不只靠 cwd**：cwd 是隐式耦合——dsh 从别处起（或 cwd 不是 ML 仓根）时，
host 半会去读**另一个**数据根的文件，甚至读到上一次会话的**残留**描述符
（排查真机问题时实测到：面板拿到一个早已退出的实例的端口）。显式环境变量把
"我的数据根在哪"讲清楚。

## 构建（一次性）

`lib/` 是构建产物且**不入库**（与 EL 同约定）。首次使用或改了 `src/` 后：

```powershell
cd dsh
npm install          # 只装 tsdown/typescript（react 等是 dsh 提供的 peer，外置）
npm run bundle       # → lib/index.mjs（host 半）+ lib/client.js（client 半）
```

> ⚠ `lib/client.js` 必须保持 dsh 的 `__ModuleLoader__.load({id, factory})`
> classic-script 格式（`tsdown.config.js` 的 banner/intro/footer 负责）。
> 直接写普通 ESM 会让 dsh 的合并包整体 SyntaxError。
> banner 里的 `id` 必须等于 `package.json` 的 `name`（`ml-toolbox-dsh`）。

没构建也能用：`launcher.py` 只在 `lib/index.mjs` 存在时才追加本插件 overlay，
否则打一条"面板插件未构建"的降级日志——AI 模式照常，只是没有右栏面板。

> ⚠ **改了 `src/` 之后必须 `npm run bundle` 并重启 dsh**：client 半是浏览器里的
> 合并包，dsh 不会热加载它。面板行为没变时，先怀疑"bundle 没重建 / dsh 没重启"。

## 怎么被加载

`launcher.py` 起 dsh 时带**三份** overlay：

```powershell
dsh --profile web --patch ./dsh/cordis.patch.yml `
                --patch ./dsh/cordis.project.patch.yml `
                --patch runtime/dsh-ml-mcp.patch.yml --port 3081
```

- `dsh/cordis.patch.yml`（**入库**）：加载本插件（`name: './lib/index.mjs'`，相对本文件）。
- `dsh/cordis.project.patch.yml`（**入库**）：**项目隔离**——还原 webserver 的
  host/port 表达式，让 `--port` 生效。**不加它就会去抢官方实例的 `:3080`**
  （共享 profile 把 port 钉成字面量，而同 id 的 patch 是整块替换 config）。
- `runtime/dsh-ml-mcp.patch.yml`（**生成物，不入库**）：每次启动按实际 MCP 端口重写，
  内含 `@deepseek-ai/dsh-mcp-client` 条目（`- insert:` 包住）。

环境变量：`MLTB_ML_ROOT` = ML 仓根（launcher 显式设置，见上一节）。

## 参数化：监控地址从哪来

`MONITOR_URL` **不是**硬编码常量（EL 把 `http://127.0.0.1:8767` 写死在两端，两处一漂移
就是"面板空白但没人报错"的假绿）。ML 的链路只有一条：

```
权威启动 → 写 .ml-monitor-port + .ml-mecha-runtime.json（单一真源）
   → client 半的 tab body（session 作用域 slot）
   → inject 的 resolveMonitorBase() → 同源 fetch('/ml-toolbox/monitor-url')
   → host 半读描述符（根 = MLTB_ML_ROOT 或 cwd）→ 返回 {base: "http://127.0.0.1:<实际端口>"}
```

读不到就**如实显示"监控端点未知"+ 重试**，绝不回落某个默认端口。

## R1 观测程序：dsh 内置客户端是否命中"HTTP 世代死区"

**为什么观测**：dsh 官方 README 自列——"HTTP 建立连接后，请求失败使用 SDK 传输的
恢复机制，而非重新创建连接"，dev note 更直说"Streamable HTTP 的重连归属仍未决定"。
EL 正是为这条死区写了自愈桥。ML 目前不写桥，所以必须知道**它到底会不会咬人**。

**触发 P3（上自愈桥）的判据**：下面任一成立。

### 手动步骤（约 3 分钟）

1. 起 AI 模式：`python launcher.py` → 选「AI 模式」（或直接
   `python app_entry.py --authority` 后手工起 dsh）。
2. 在 dsh 里让 AI 调一个**只读**工具（例如问"现在有哪些方法能用？"）→ 确认能拿到结果。
   记下此时 `runtime/.ml-mecha-runtime.json` 里的 `mcp_port`（记为 P1）。
3. **在 dsh 不退出的前提下**杀掉权威进程：
   `Get-Process python | Where-Object { $_.Id -eq <authority_pid> } | Stop-Process -Force`
   （`authority_pid` 见描述符）。
4. 重新起一台权威（同一 root）：`python app_entry.py --authority`
   → 它会拿到一个**新端口**（记为 P2 ≠ P1，因为 MCP 是 OS 分配）。
5. **不要重启 dsh**。回到 dsh 会话里，再让 AI 调同一个只读工具。

### 观察点

| 观察 | 含义 | 动作 |
|---|---|---|
| 工具调用**成功** | 内置客户端已按新端点/新 session 恢复（或 dsh 侧重连生效） | 不上 P3；把这次结果记进 `07` 的 R1 状态 |
| 工具**仍在列表里，但每次调用都失败**（报 404 / session not found / 连接类错误），**重启 dsh 后立刻恢复** | **命中死区**（旧 `Mcp-Session-Id` 只抛错、不触发 `onclose`） | **上 P3**（移植 EL `dsh/src/host/mcp-bridge.ts`） |
| 工具**从列表里消失**了 | `reconnect.maxAttempts`（默认 10）耗尽后工具被移除、重连停止 | 说明 supervisor 侧接管了 HTTP 世代；不算死区，但要写进故障排查 |
| dsh 自己崩了/日志报插件加载失败 | 与 R1 无关 | 查 dsh 控制台输出与 `dsh --profile web --patch ... --dump-config` |

> 也可以先用更便宜的近似观察：只杀权威、**不**重启（第 4 步跳过），此时"工具在但恒失败"
> 是**预期**（服务真的没了）。要区分"服务没了"和"死区"，**必须**做第 4 步重启权威。

## 安全边界（红线）

- 本插件的 host 半**只注册一个查询命令**，不 spawn/kill 进程、不写任何文件。
- 监控端点是**只读**的（`POST` 一律 404），绑回环、无鉴权——理由见
  `ml_mecha/monitor_http.py` 的模块 docstring（读的是操作流水，不是域数据覆写面）。
- 切回 GUI 模式在 P2 走**看门人控制台**；dsh 里的「→ GUI」按钮 + `/ml-gui` 命令属 P3。
