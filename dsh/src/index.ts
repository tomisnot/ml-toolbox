/**
 * ML Toolbox dsh 插件 —— **HOST 半**（Cordis 插件，函数形态）。
 *
 * 职责**只有一件**：把 ML 只读监控端点的地址交给 client 半（面板要轮询它）。
 *
 * ## 现在这份实现来自共享资产
 *
 * 路由处理、端口文件解析、"绝不回落"的纪律都在逐字复制的
 * `../dsh-panel/monitor-url.ts`（`makeMonitorUrlHandler` + `resolveMonitorBase`）里；
 * 本文件只剩 ML 的两件事：**参数块**（`../dsh-panel/panel-config.ts`）与
 * **数据根从哪来**（`MLTB_ML_ROOT` 环境变量，看门人起 dsh 时显式设置）。
 *
 * ## ⚠ 行为变更（相对迁移前，已申报）
 *
 * **地址来源：两级 → 单级。** 迁移前本文件**先读 `.ml-mecha-runtime.json` 的
 * `monitor_port`，读不到再回落裸端口文件**；资产纪律是**单级、绝不回落**。
 * ⇒ 现在**只读 `.ml-monitor-port`**（`PANEL_CONFIG.PORT_FILE`），描述符**不再参与取址**。
 * 后果（刻意）：裸端口文件缺失而描述符里有端口时，旧版能出面板、**新版回 `503` + 可读错误**
 * ——"权威没起来"不许伪装成"连上了但空面板"。
 *
 * ## 为什么是同源只读 HTTP 路由，而不是一个 command
 *
 * 早先注册 `/ml-monitor-url` command，由 client 经 `remote.commands.execute(...)` 取地址。
 * 真机实测挂在两处：① command 返回类型是 `RemoteResult<CommandExecution>`，地址在
 * `value.result.text`，按 `value.text` 读永远取不到；② 每次取地址都会往会话里写
 * `command/run` + `command/done`，污染聊天记录。现在走同源 `fetch`：无包装、无跨源、
 * 无会话副作用。
 *
 * 安全红线：本插件**只读** + 只暴露一个"地址查询"路由，**不** spawn/kill 任何进程、
 * 不写任何文件。不移植 EL 的自愈 MCP 桥（ML 的工具接入走 dsh 内置
 * `@deepseek-ai/dsh-mcp-client`；自愈桥只在命中 R1 时上，见 `dsh/README.md`）。
 */
import { PANEL_CONFIG } from './dsh-panel/panel-config.ts'
import { makeMonitorUrlHandler } from './dsh-panel/monitor-url.ts'

/** Cordis 插件显示名（诊断用）。 */
export const name = 'ml-toolbox-monitor'

/** 只依赖 webserver 服务（注册那条只读路由）。 */
export const inject = ['webServer']

/** 挂载路径（**绝对、无尾斜杠**，dsh 路由契约要求）——与参数块同源，只有一处字面量。 */
export const ROUTE_PATH = PANEL_CONFIG.ROUTE_PATH

/**
 * 数据根的环境变量名：**看门人起 dsh 时显式设置**，指向 ML 仓根。
 *
 * 为什么不只靠 `process.cwd()`：cwd 是隐式耦合——dsh 从别处起（或 cwd 不是仓根）时，
 * 本半会去读**另一个**数据根的端口文件，甚至读到上一次会话的**残留**（真机排查时实测到过）。
 * 显式环境变量把"我的数据根在哪"讲清楚；cwd 只作兜底（资产 `resolveMonitorBase` 的默认）。
 */
export const ML_ROOT_ENV = 'MLTB_ML_ROOT'

/** 本插件该读哪个数据根：显式环境变量优先；没设就交给资产用 `process.cwd()`。 */
export function pluginRoot(): string | undefined {
  const fromEnv = process.env[ML_ROOT_ENV]
  return fromEnv && fromEnv.trim() ? fromEnv.trim() : undefined
}

/**
 * 路由处理：能解析就 `200 {base}`；解析不了就 `503` + 可读错误（**不回落**）。
 *
 * ⚠ **每次请求现读端口文件**（资产实现，不缓存）：权威重启换端口后下一拍就落在新端口上。
 */
export const monitorUrlHandler = makeMonitorUrlHandler({
  get root() { return pluginRoot() },
  portFile: PANEL_CONFIG.PORT_FILE,
})

/** 挂载只读路由（路径来自参数块）。 */
export async function apply(ctx: any): Promise<void> {
  await ctx.effect(() => {
    const dispose = ctx.webServer.register({
      kind: 'exact',
      path: ROUTE_PATH,
      handler: monitorUrlHandler,
    })
    return () => {
      try {
        dispose()
      } catch {
        /* ignore */
      }
    }
  }, 'ml-toolbox-monitor: monitor endpoint address route')
}
