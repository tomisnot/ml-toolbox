/**
 * ML Toolbox dsh 插件 —— **CLIENT 半**（浏览器侧 Cordis 插件）。
 *
 * 三件事（移植自 EL `dsh/src/client/`，已按 ML 改名/参数化）：
 *  1. 会话头部「◈ 监控」按钮 → 开右栏 + 打开 `ml-toolbox-monitor` tab。
 *  2. 右栏 tab body（`MonitorTabBody`）→ 轮询 ML 权威的**只读**监控端点，
 *     三页：飞行记录仪（写事件）/ 配置态 / 运行记录。
 *  3. **监控地址参数化**：`MONITOR_URL` 不再是硬编码常量——经注入的
 *     `resolveMonitorBase()` 去取 dsh **同源**只读路由 `/ml-toolbox/monitor-url`
 *     （host 半读权威写的运行期描述符后返回 `{base}`）。
 *
 * ⚠ **为什么不再走 command**（真机教训）：早先经
 * `remote.commands.execute(sessionId, '/ml-monitor-url')` 取地址，挂了两处——
 * ① command 的返回类型是 `RemoteResult<CommandExecution>`，地址在
 * `value.result.text`，按 `value.text` 读永远取不到（真机面板显示"监控端点未知"）；
 * ② 每次取地址都会往会话里写 `command/run` + `command/done`，污染聊天记录。
 * 现在走同源 `fetch`：无包装、无跨源、无会话副作用。
 *
 * ⚠ 与 EL 的另一处有意差异：EL 把 `http://127.0.0.1:8767` 硬编码在**两端**
 * （端点侧默认参数 + 面板侧常量），两处一漂移就是"面板空白但没人报错"的假绿。
 * ML 只有一个来源（权威写的运行期描述符），由 host 半经同源路由转出。
 *
 * ⚠ 本目录经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改源码后须
 * `npm run bundle` 并**重启 dsh** 才生效（dsh 的 web client 只吃合并包）。
 */
import type { Context } from '@deepseek-ai/cordis'
import { MonitorTabBody } from './MonitorTabBody'
import { CockpitButton, type CockpitInjected } from './buttons'

/** Cordis 插件名（与 host 半、package.json 一致）。 */
export const name = 'ml-toolbox-monitor'

/** slots（按钮 + tab body）+ layout（开右栏）+ 右栏 tab 服务。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs']

/** 右栏 tab 的 kind/id（改名自 EL 的 `re0-cockpit`）。 */
const TAB_ID = 'ml-toolbox-monitor'

/** host 半注册的同源只读路由（地址查询）。 */
export const ADDRESS_ROUTE = '/ml-toolbox/monitor-url'

/** tab body 的注入面：解析监控端点基址（由 host 半的同源路由提供）。 */
export interface MonitorInjected {
  resolveMonitorBase(): Promise<string>
}

/**
 * 取监控端点基址：同源 `fetch` → `{base}`。
 *
 * 任何一步不对都**抛错**（面板会显示"监控端点未知：…"）——不回落到任何默认端口。
 * 负例见判据：喂旧的 command 返回形状（`{ok, value}`）必须失败。
 *
 * ``doFetch`` 可注入，方便判据在 Node 里喂真形状与各种失败形状。
 */
export async function fetchMonitorBase(
  doFetch: typeof fetch = fetch,
): Promise<string> {
  const res = await doFetch(ADDRESS_ROUTE, { cache: 'no-store' })
  if (!res.ok) throw new Error(`地址路由返回 ${res.status}`)
  const data: any = await res.json()
  const base = data && typeof data.base === 'string' ? data.base : ''
  if (!base) throw new Error('地址路由的返回里没有 base（形状不对？）')
  return base.trim().replace(/\/+$/, '')
}

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // 会话头部：「◈ 监控」按钮（开右栏 + 打开我们的 tab）。
    const disposeCockpit = ctx.slots.inject(
      'conversation.session.header.actions',
      () =>
        ctx.slots.register(
          {
            name: 'conversation.session.header.actions',
            id: TAB_ID,
            order: 200,
            label: '◈ 监控',
            inject: (): CockpitInjected => ({
              openCockpit: () => {
                ctx.layout?.openRightbar?.(true, false)
                try {
                  ;(ctx as any).sidebarRight?.openTab?.(TAB_ID)
                } catch {
                  /* 降级：右栏开了但 tab 没切过去，不影响面板本体 */
                }
              },
            }),
          },
          CockpitButton,
        ),
    )

    // 右栏 tab 类型 + body。`sidebar.right.pane.tab` 是 session 作用域 slot，
    // inject 会拿到框架解析好的 sessionId——本插件用不到它（地址走同源路由），
    // 保留形参只是匹配 slot 契约。
    const releaseTab = (ctx as any).sidebarRightTabs?.register?.({
      id: TAB_ID,
      kind: TAB_ID,
      title: () => 'ML 监控',
    }) || null
    const disposeBody = ctx.slots.inject('sidebar.right.pane.tab', () =>
      ctx.slots.register(
        {
          name: 'sidebar.right.pane.tab',
          key: TAB_ID,
          inject: (): MonitorInjected => ({ resolveMonitorBase: fetchMonitorBase }),
        } as any,
        MonitorTabBody as any,
      ))

    return () => {
      if (disposeBody) {
        try {
          disposeBody()
        } catch {
          /* ignore */
        }
      }
      if (typeof releaseTab === 'function') {
        try {
          releaseTab()
        } catch {
          /* ignore */
        }
      }
      try {
        disposeCockpit?.()
      } catch {
        /* ignore */
      }
    }
  }, 'ml-toolbox-monitor: client monitor UI')
}
