/**
 * ML Toolbox dsh 插件 —— **CLIENT 半**（浏览器侧 Cordis 插件）。
 *
 * ## ⭐ 2026-10 阶段②：本文件 = **一行 bootstrap + 自定义 UI**（别的都不写了）
 *
 * ```ts
 * export { apply, name, inject } from '@mecha/dsh-panel/panel/auto.ts'   // ← 框架的代码
 * ```
 *
 * 那一行接上**框架自己的 client 插件**（`@mecha/dsh-panel/panel/auto.ts`）：它读框架起的
 * dsh 时按**声明**（`ml_mecha/mecha_site.py` 的 `Service.monitor`）生成的那份参数，
 * **按服务挂 N 个 tab**（三个项目各挂自己那一份），头部按钮**恒为 1 个**。
 * ⚠ 为什么**不是**裸 `import`：cordis 只认**模块自己的** `name`/`inject`/`apply`
 * —— 裸 import 只执行模块体、`apply` 不会被调；且资产 `sideEffects:false` 还会摇掉它。
 *
 * ## 本文件还剩什么：**只有"自定义 UI"**（§7.3.3 定论：保留为可选）
 *
 * 声明是**数据**，装不下"怎么取值 / 怎么格式化"那种**代码**。ML 的主指标列要按来源各取
 * 一次（操作史行在 `score`、磁盘行在 `metrics[主指标]`；旧自绘面板只读 `r.value`，
 * **磁盘行丢过值**）⇒ 那份**呈现细化**留在本项目的 `extraPages.ts`，在这里注册。
 * ⚠ 分工不可重叠：**页集/路由/文案在声明里**（一处真值），这里只细化**呈现**。
 *
 * ## 已删掉的东西（别往回加）
 *
 * - 会话头部「◈ 监控」按钮：阶段① 起归框架（全生态**唯一一个**，语义 = 开右栏）；
 * - `sidebarRightTabs` / `sidebar.right.pane.tab` 的手写注册：同上，且
 *   `title: () => panelConfig().title` 是**渲染时求值**的 thunk ⇒ 三家装进同一个 dsh 时
 *   三个 tab 的标题与取数路由**全落到"最后注册那一份"**（阶段① 修掉的真缺陷）；
 * - 本仓的面板参数常量（**已删**）：面板参数的家已经在声明里（`mecha_site.py`）；
 *   ⭐ **2026-10-09 起 `ml-panel-params.ts` 也删了**（连同 host 半那条监控地址路由一起
 *   上提进框架）：地址路由由**框架的 host 插件**按 `MECHA_PANEL_PORT_FILES` 现取端口
 *   ⇒ 本仓**不再有**参数的第二处真值，也没有"两处必须同值"这回事了。
 *
 * ⚠ 本目录经 tsdown 打成 CJS + `__ModuleLoader__.load` 包装；改源码后须
 * `npm run bundle` 并**重启 dsh** 才生效（dsh 的 web client 只吃合并包）。
 */
import {
  apply,
  name,
  inject,
  contributeExtraPageRender,
  extraPageRenderKey,
} from '@mecha/dsh-panel/panel/auto.ts'
import { ML_EXTRA_PAGE_RENDERS } from './extraPages.ts'

// ⭐ **bootstrap**：把框架那份插件的 `apply/name/inject` 交给 dsh（本项目这一份 bundle）。
export { apply, name, inject }

// ---- 以下 = **自定义 UI**（本项目的附加页呈现细化）----
// ⚠ 模块加载期注册（早于 dsh 调 `apply`）⇒ 框架挂 tab 时拿得到。
// ⚠ 键里必须带服务 id（页 id 只在本服务内唯一）；收尾器本文件不留 —— 它的寿命与
//   本 bundle 一致（浏览器里本模块不会卸载，除非整页重载）。
for (const [pageId, render] of Object.entries(ML_EXTRA_PAGE_RENDERS)) {
  contributeExtraPageRender(extraPageRenderKey('ML', pageId), render)
}
