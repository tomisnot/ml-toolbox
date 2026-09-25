/**
 * ML Toolbox dsh 插件 —— **HOST 半**（Cordis 插件，函数形态）。
 *
 * 职责**只有一件**：把 ML 只读监控端点的地址交给 client 半（驾驶舱面板要轮询它）。
 *
 * ## 为什么是同源只读 HTTP 路由，而不是一个 command
 *
 * 早先的版本注册 `/ml-monitor-url` command，由 client 经
 * `remote.commands.execute(sessionId, …)` 取地址。真机实测挂在两处：
 *
 * 1. **返回值形状脆弱**：command 的返回类型是
 *    `RemoteResult<CommandExecution | undefined>`，其中
 *    `CommandExecution = { commandId, result: CommandResult }`、
 *    `CommandResult = {kind, text}` ⇒ 地址在 `r.value.result.text`。当时按
 *    `r.value.text` 读 ⇒ 永远取不到，面板显示"监控端点未知"。（当时的自测把
 *    `r.value` 按错形状 stub 了，所以自测绿、真机必挂——自证假绿。现在的判据
 *    一律喂**真形状**，并加一条"喂旧形状必须失败"的负例。）
 * 2. **取一次地址会污染会话**：每次 `commands.execute` 都会往会话里写
 *    `command/run` + `command/done` 两条记录。面板只是想知道一个端口，
 *    不该在聊天记录里留痕。
 *
 * 改为挂在 dsh **同源** webserver 上的只读路由（面板直接 `fetch` 同源路径：
 * 无跨源、无会话副作用、无返回值包装）。dsh 的契约是
 * `webServer.register({kind:'exact', path:'/absolute/no/trailing/slash', handler})`，
 * handler **拥有完整响应生命周期**，返回 disposer。
 *
 * ## 单一来源不变
 *
 * 端口仍由权威写进 `.ml-monitor-port` / `.ml-mecha-runtime.json`；本半只读它。
 * **读不到就如实报错（非 200 + 可读 JSON），绝不回落硬编码端口**——回落一个
 * "常见端口"等于把面板悄悄指向别人的服务。
 *
 * 不移植 EL 的自愈 MCP 桥（`dsh/src/host/mcp-bridge.ts`）：ML 的工具接入走 dsh
 * 内置 `@deepseek-ai/dsh-mcp-client`。自愈桥只在命中 R1 时上（见 `dsh/README.md`
 * 的观测程序与 `docs/mecha/07-...md` §4 P3）。
 *
 * 安全红线：本插件**只读** + 只暴露一个"地址查询"路由，**不** spawn/kill 任何
 * 进程、不写任何文件（切回 GUI 在 P2 走看门人控制台；`/ml-gui` 按钮属 P3）。
 */
import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { join } from 'node:path'

/** Cordis 插件显示名（诊断用）。 */
export const name = 'ml-toolbox-monitor'

/** 只依赖 webserver 服务（注册那条只读路由）。 */
export const inject = ['webServer']

/** 挂载路径（**绝对、无尾斜杠**，dsh 路由契约要求）。 */
export const ROUTE_PATH = '/ml-toolbox/monitor-url'

/** 运行期描述符与裸端口文件名（与 ml_mecha.runtime 的常量一致）。 */
const RUNTIME_JSON = '.ml-mecha-runtime.json'
const MONITOR_PORT_FILE = '.ml-monitor-port'

/**
 * 数据根的环境变量名：**看门人起 dsh 时显式设置**，指向 ML 仓根。
 *
 * 为什么不只靠 `process.cwd()`：cwd 是隐式耦合——dsh 从别处起（或 cwd 不是仓根）
 * 时，本半会去读**另一个**数据根的运行期文件，甚至读到上一次会话的**残留**
 * 描述符（真机排查时实测到过：面板拿着一个早已退出的实例的端口）。显式环境变量
 * 把"我的数据根在哪"讲清楚；cwd 只作为兜底。
 */
export const ML_ROOT_ENV = 'MLTB_ML_ROOT'

/**
 * 解析监控端点基址（`http://127.0.0.1:<port>`）。
 *
 * 优先级：运行期描述符（**单一真源**，含 mode/端口/pid）> 裸端口文件。
 * 两者都读不到就抛错——由路由转成给用户看的失败 JSON。
 */
export function resolveMonitorBase(root: string = process.cwd()): string {
  try {
    const raw = readFileSync(join(root, RUNTIME_JSON), 'utf8')
    const port = Number(JSON.parse(raw)?.monitor_port)
    if (Number.isInteger(port) && port > 0) return `http://127.0.0.1:${port}`
  } catch {
    /* 落到下一个来源；在这里不报错，让下面的分支给出更准确的原因 */
  }
  let raw = ''
  try {
    raw = readFileSync(join(root, MONITOR_PORT_FILE), 'utf8').trim()
  } catch (err) {
    throw new Error(
      `读不到监控端口（${MONITOR_PORT_FILE} 不存在，${RUNTIME_JSON} 里也没有 ` +
        `monitor_port）：软件没在 AI 模式跑？(${(err as Error).message})`,
    )
  }
  if (!/^\d+$/.test(raw)) {
    throw new Error(`监控端口文件内容非法：${JSON.stringify(raw)}`)
  }
  return `http://127.0.0.1:${raw}`
}

/** 本插件该读哪个数据根：显式环境变量优先，cwd 兜底。 */
export function pluginRoot(): string {
  const fromEnv = process.env[ML_ROOT_ENV]
  return fromEnv && fromEnv.trim() ? fromEnv.trim() : process.cwd()
}

/** 写一份 JSON 响应（handler 拥有完整响应生命周期，故自己收尾）。 */
function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = Buffer.from(JSON.stringify(payload), 'utf8')
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.setHeader('Content-Length', String(body.length))
  res.setHeader('Cache-Control', 'no-store')
  res.end(body)
}

/** 路由处理：能解析就 200 + `{base}`；解析不了就非 200 + 可读错误。 */
export function monitorUrlHandler(_req: IncomingMessage,
                                  res: ServerResponse): void {
  try {
    sendJson(res, 200, { base: resolveMonitorBase(pluginRoot()) })
  } catch (err) {
    sendJson(res, 503, { error: `监控端点未知：${(err as Error).message}` })
  }
}

/** 挂载只读路由 `/ml-toolbox/monitor-url`。 */
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
