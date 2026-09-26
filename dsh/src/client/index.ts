/**
 * ML Toolbox dsh 插件 —— **CLIENT 半**（浏览器侧 Cordis 插件）。
 *
 * 三件事：
 *  1. 会话头部「◈ 监控」按钮 → 开右栏 + 打开 `ml-toolbox-monitor` tab。
 *  2. 右栏 tab body = **共享资产的面板**（`../dsh-panel/MonitorTabBody.tsx`，逐字复制自
 *     `mecha/dsh-panel`）；本仓**不再自己渲染**（旧的自绘面板已删——用户已裁"ML 也并入"，
 *     三家共用一份面板）。
 *  3. **ML 的附加页用声明式列表挂上**（`ExtraPageSpec[]`）：领域词（列名/文案/取值）全留在
 *     本文件，资产里零 ML 字面量。
 *
 * ## 取数走资产的管线（本文件**不再**自己 fetch）
 *
 * 地址、四基础路由取数、三档错误态（`address`/`offline`/`route`）、`empty` 档、R8 自证全在
 * `../dsh-panel/` 里；本文件只声明"ML 多出来的那两页"。
 *
 * ## ⚠ 两条已申报的行为变更（相对旧的自绘面板）
 *
 * 1. **一页并列 → 两页**：旧面板把「操作史运行记录（`/summary.recent_runs`）」与
 *    「磁盘存档（`/runs`）」画在**同一页**；资产的附加页机制是**一页一条路由** ⇒ 拆成
 *    「运行记录」与「磁盘存档」两页（功能不丢，位置变）。
 * 2. **布局与 CSS 取资产的**（`mltb-` 前缀那套自绘样式随旧面板一起删除）。
 *
 * ⚠ 本目录经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改源码后须
 * `npm run bundle` 并**重启 dsh** 才生效（dsh 的 web client 只吃合并包）。
 */
import type { Context } from '@deepseek-ai/cordis'
import { MonitorTabBody } from '../dsh-panel/MonitorTabBody.tsx'
import type { ExtraPageSpec } from '../dsh-panel/panel-view.ts'
import { PANEL_CONFIG } from '../dsh-panel/panel-config.ts'
import { CockpitButton, type CockpitInjected } from './buttons'
import { EXTRA_PAGES } from './extraPages.ts'

/** Cordis 插件名（与 host 半、package.json 一致）。 */
export const name = 'ml-toolbox-monitor'

/** slots（按钮 + tab body）+ layout（开右栏）+ 右栏 tab 服务。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs']

/** 右栏 tab 的 kind/id。 */
const TAB_ID = 'ml-toolbox-monitor'

/**
 * tab body 的注入面：**共享面板自己要的只有 `extraPages`**（地址与取数它自己走资产管线）。
 * slot 的 `inject()` 返回值就是组件的 props ⇒ 直接把资产组件注册进去即可，不需要包装器
 * （包装器会把 hooks 边界挪一层，且是多余的）。
 */
export interface MonitorInjected {
  extraPages: ExtraPageSpec[]
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
                  ctx.sidebarRight?.openTab?.(TAB_ID)
                } catch {
                  /* 降级：右栏开了但 tab 没切过去，不影响面板本体 */
                }
              },
            }),
          },
          CockpitButton,
        ),
    )

    // 右栏 tab 类型 + body（`sidebar.right.pane.tab` 是 session 作用域 slot）。
    const releaseTab = ctx.sidebarRightTabs?.register?.({
      id: TAB_ID,
      kind: TAB_ID,
      title: () => PANEL_CONFIG.TITLE,
    }) || null
    const disposeBody = ctx.slots.inject('sidebar.right.pane.tab', () =>
      ctx.slots.register(
        {
          name: 'sidebar.right.pane.tab',
          key: TAB_ID,
          inject: (): MonitorInjected => ({ extraPages: EXTRA_PAGES }),
        },
        MonitorTabBody as never,
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
