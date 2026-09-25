/**
 * ML Toolbox dsh 插件 —— **CLIENT 半**（浏览器侧 Cordis 插件）。
 *
 * 三件事（移植自 EL `dsh/src/client/`，已按 ML 改名/参数化）：
 *  1. 会话头部「◈ 监控」按钮 → 开右栏 + 打开 `ml-toolbox-monitor` tab。
 *  2. 右栏 tab body（`MonitorTabBody`）→ 轮询 ML 权威的**只读**监控端点，
 *     三页：飞行记录仪（写事件）/ 配置态 / 运行记录。
 *  3. **监控地址参数化**：`MONITOR_URL` 不再是硬编码常量——经注入的
 *     `resolveMonitorBase()` 调 host 半的 `/ml-monitor-url` command 取，
 *     地址来自权威写的运行期描述符（每次启动端口都可能不同）。
 *
 * ⚠ 与 EL 的两处有意差异：
 *  - EL 把 `http://127.0.0.1:8767` 硬编码在**两端**（端点侧默认参数 + 面板侧常量），
 *    两处一漂移就是"面板空白但没人报错"的假绿。ML 只有一个来源（描述符）。
 *  - EL 的「→ GUI」按钮（`/re0-gui`）**不在本轮**：切回 GUI 在 P2 走看门人控制台，
 *    按钮属 P3（见 `docs/mecha/07-...md` §4）。
 *
 * ⚠ 本目录经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改源码后须
 * `npm run bundle` 并重启 dsh 才生效（dsh 的 web client 只吃合并包）。
 */
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { Context } from '@deepseek-ai/cordis'
import { MonitorTabBody } from './MonitorTabBody'
import { CockpitButton, type CockpitInjected } from './buttons'

/** Cordis 插件名（与 host 半、package.json 一致）。 */
export const name = 'ml-toolbox-monitor'

/** slots（按钮 + tab body）+ layout（开右栏）+ 右栏 tab 服务 + remote.commands。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs',
                       'remote', 'remote.commands']

/** 右栏 tab 的 kind/id（改名自 EL 的 `re0-cockpit`）。 */
const TAB_ID = 'ml-toolbox-monitor'

/** tab body 的注入面：解析监控端点基址（由 host 半命令提供，非硬编码）。 */
export interface MonitorInjected {
  resolveMonitorBase(): Promise<string>
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

    // 右栏 tab 类型 + body。`sidebar.right.pane.tab` 是 **session 作用域**的
    // slot（dsh slots 契约），因此 inject 会拿到框架解析好的 sessionId——
    // 面板正是靠它去发 /ml-monitor-url。
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
          inject: (sessionId: SessionId): MonitorInjected => ({
            resolveMonitorBase: async () => {
              const r = await ctx.remote!.commands!.execute(
                sessionId, '/ml-monitor-url', [])
              if (!r.ok) throw new Error(r.error.message)
              // `r.value` 的形状未由 EL 钉死（它只判 undefined）⇒ 防御性提取，
              // 拿不到就抛错让面板显示"地址未知"，绝不回落硬编码端口。
              const value: any = r.value
              const text = typeof value === 'string'
                ? value
                : (value && typeof value.text === 'string' ? value.text : '')
              if (!text) throw new Error('命令未返回监控地址')
              return text.trim().replace(/\/+$/, '')
            },
          }),
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
