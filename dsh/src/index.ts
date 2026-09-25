/**
 * ML Toolbox dsh 插件 —— **HOST 半**（Cordis 插件，函数形态）。
 *
 * 职责**只有一件**：把 ML 只读监控端点的地址告诉 client 半（驾驶舱面板要轮询它）。
 *
 * 为什么需要这一跳：client 半跑在**浏览器**里，读不了文件、也不能直接调 MCP 工具
 * （dsh 事实：client 只能发 command）；而监控端口是**每次启动由系统分配**的
 * （写进权威的运行期描述符）。所以由 host 半（Node，同机、cwd = 仓根）读那个
 * 描述符，再经 `/ml-monitor-url` command 回给 client。
 *
 * **没有硬编码默认值**：读不到就如实报错。回落一个"常见端口"（例如 EL 的 8767）
 * 等于把面板悄悄指向别人的服务——那正是 P2c 要删掉的东西。
 *
 * 不移植 EL 的自愈 MCP 桥（`dsh/src/host/mcp-bridge.ts`）：ML 的工具接入走 dsh
 * 内置 `@deepseek-ai/dsh-mcp-client`。自愈桥只在命中 R1 时上（见 `dsh/README.md`
 * 的观测程序与 `docs/mecha/07-...md` §4 P3）。
 *
 * 安全红线：本插件**只读** + 只留一个"地址查询"命令，**不** spawn/kill 任何进程、
 * 不写模式文件（切回 GUI 在 P2 走看门人控制台；`/ml-gui` 按钮属 P3）。
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

/** Cordis 插件显示名（诊断用）。 */
export const name = 'ml-toolbox-monitor'

/** 只要 commands 服务（注册 /ml-monitor-url）。 */
export const inject = ['commands']

/** 运行期描述符与裸端口文件名（与 ml_mecha.runtime 的常量一致）。 */
const RUNTIME_JSON = '.ml-mecha-runtime.json'
const MONITOR_PORT_FILE = '.ml-monitor-port'

/**
 * 解析监控端点基址（`http://127.0.0.1:<port>`）。
 *
 * 优先级：运行期描述符（**单一真源**，含 mode/端口/pid）> 裸端口文件。
 * 两者都读不到就抛错——由 command 转成给用户看的失败文本。
 */
export function resolveMonitorBase(root: string = process.cwd()): string {
  try {
    const raw = readFileSync(join(root, RUNTIME_JSON), 'utf8')
    const port = Number(JSON.parse(raw)?.monitor_port)
    if (Number.isInteger(port) && port > 0) return `http://127.0.0.1:${port}`
  } catch {
    /* 落到下一个来源；不在这里报错，让下面的分支给出更准确的原因 */
  }
  let raw = ''
  try {
    raw = readFileSync(join(root, MONITOR_PORT_FILE), 'utf8').trim()
  } catch (err) {
    throw new Error(
      `读不到监控端口（${MONITOR_PORT_FILE} 不存在，${RUNTIME_JSON} 里也没有 monitor_port）：` +
        `软件没在 AI 模式跑？(${(err as Error).message})`,
    )
  }
  if (!/^\d+$/.test(raw)) {
    throw new Error(`监控端口文件内容非法：${JSON.stringify(raw)}`)
  }
  return `http://127.0.0.1:${raw}`
}

/** 挂载 `/ml-monitor-url`：把只读监控端点地址回给 client 半。 */
export async function apply(ctx: any): Promise<void> {
  await ctx.effect(() => {
    const dispose = ctx.commands.register({
      name: 'ml-monitor-url',
      description: '返回 ML 只读监控端点地址（驾驶舱面板轮询用）',
      handler: async () => {
        try {
          return { kind: 'success' as const, text: resolveMonitorBase() }
        } catch (err) {
          return { kind: 'error' as const,
                   text: `监控端点未知：${(err as Error).message}` }
        }
      },
    })
    return () => {
      try {
        dispose()
      } catch {
        /* ignore */
      }
    }
  }, 'ml-toolbox-monitor: monitor endpoint address command')
}
