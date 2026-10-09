/**
 * ML Toolbox dsh 插件 —— **CLIENT 半**（浏览器侧 Cordis 插件）。
 *
 * 两件事（⚠ 按钮**不在这里**了，见下）：
 *  1. 右栏 tab body = **共享资产的面板**（`@mecha/dsh-panel/panel/MonitorTabBody.tsx`，
 *     `file:` 依赖 ⇒ junction 指向 mecha 仓那一份，磁盘上只有一份代码）；本仓**不再自己渲染**。
 *  2. **ML 的附加页用声明式列表挂上**（`ExtraPageSpec[]`）：领域词（列名/文案/取值）全留在
 *     本文件，资产里零 ML 字面量。
 *
 * ## ⭐ 2026-10 阶段①：改用框架 helper（**按钮与 tab 注册都交给框架**）
 *
 * 原先本文件自己做三件易错的事：① 注册会话头部「◈ 监控」按钮；② 注册 `sidebarRightTabs`；
 * ③ 注册正文字段。三家里**各写一遍** ⇒ 两个已实测的后果：
 * - **每家一个按钮** ⇒ 3 个项目 = 3 个「◈ 监控」（用户要的是"唤醒**所有**监控面板"）；
 * - **`title: () => panelConfig().title` 渲染时才求值** ⇒ 三家装进同一个 dsh 时
 *   **三个 tab 的标题与取数路由全落到"最后注册那一份"**（且不报错）。
 *
 * ⇒ 现在只调一次 `mountClientPanel`：参数**跟着这一次注册走**（标题闭包捕获自己那份、
 * 正文以显式 prop 收到自己那份），头部按钮由框架注册**唯一一个**（语义 = 开右栏）。
 * ⚠ 本文件仍**必须把 `panelConfig` 交给它**（那是 ML 专有的值，资产不许猜）。
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
import { mountClientPanel } from '@mecha/dsh-panel/panel/mount-client-plugin.ts'
import { MonitorTabBody } from '@mecha/dsh-panel/panel/MonitorTabBody.tsx'
import type { ExtraPageSpec } from '@mecha/dsh-panel/panel/panel-view.ts'
import { ML_PANEL } from '../ml-panel-params.ts'
import { EXTRA_PAGES } from './extraPages.ts'

/** Cordis 插件名（与 host 半、package.json 一致）。 */
export const name = 'ml-toolbox-monitor'

/** slots（tab body）+ layout（开右栏）+ 右栏 tab 服务。 */
export const inject = ['slots', 'layout', 'sidebarRight', 'sidebarRightTabs']

/** ⭐ §7.3.3 定案：逐服务一个 `kind` / 一个 `id`（契约拒绝重复 id 与同 tier 同 kind 二次注册）。 */
const TAB_KIND = 'mecha:ML'
const TAB_ID = 'mecha-panel:ML'

/**
 * tab body 的注入面：**共享面板自己要的只有 `extraPages`**（地址与取数它自己走资产管线）。
 * ⚠ 面板参数**不在这里**——由 `mountClientPanel` 以显式 prop 传给正文（各读各的）。
 */
export interface MonitorInjected {
  extraPages: ExtraPageSpec[]
}

export async function apply(ctx: Context): Promise<void> {
  await ctx.effect(() => {
    // ⭐ 一次调用管全部：标题/正文绑**ML 自己那份**参数；头部按钮由框架注册**唯一一个**。
    const dispose = mountClientPanel(ctx as never, {
      panelConfig: ML_PANEL,
      kind: TAB_KIND,
      id: TAB_ID,
      body: MonitorTabBody,
      bodyProps: { extraPages: EXTRA_PAGES } satisfies MonitorInjected,
      titlePrefix: '◈ 监控 · ',
    })
    return () => { dispose() }
  }, 'ml-toolbox-monitor: client monitor UI')
}
