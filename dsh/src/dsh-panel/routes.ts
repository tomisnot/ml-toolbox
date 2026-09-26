/**
 * **浏览器安全**的共享常量：默认路由 + 四基础路由表。
 *
 * ## ⚠ 为什么它必须单独一个模块（真实缺陷，2026-09-26）
 *
 * `monitor-client.ts`（**client 半**）原先从 `monitor-url.ts` 取值导入 `BASIC_ROUTES`，
 * 而后者顶层 `import { readFileSync } from 'node:fs'` ⇒ **原生消费者的浏览器 bundle
 * 里进了 `node:fs`，加载即失败**（实测：`esbuild --bundle --platform=browser` 直接
 * `Could not resolve "node:fs"`，指到 `monitor-url.ts:22`）。
 *
 * 另一个消费者（iframe 形态）当时**没事**——但那是**树摇的运气**：它恰好没用到那条链，
 * 于是整条 import 被摇掉（它自己的原话：**"我的安全是『碰巧没用那几个导出』换来的，
 * 不是结构保证的"**）。
 *
 * ⇒ 所以这里把边界**结构化**：
 * **要 `node:*` 的实现住 `monitor-url.ts`（host 半）；client 半只许 import
 * 本模块 + `monitor-client.ts` + `panel-config.ts`。**
 *
 * ## 这条边界有机械守卫（不靠打包器的聪明）
 *
 * `monitor-client.test.ts` 里有一条**导入闭包检查**：从 client 半的两个入口出发，
 * 沿相对 import 走一遍，断言闭包里**没有任何 `node:`**；并**自证闭包真的走过了若干文件**
 * （否则"没命中"只是因为什么都没读到——R8）。类型导入（`import type`）不算：
 * 它会被剥掉，不是运行时依赖。
 */

/** 中性默认路由（**不含任何项目名**）。项目可在参数块里覆盖。 */
export const DEFAULT_ROUTE_PATH = '/mecha/monitor-url'

/**
 * 面板的**四基础路由**（`mecha.cockpit` 契约里的那四条）。
 *
 * 用途：`monitor-client` 用它区分"**基础路由失败 = 权威离线**"与"**附加路由失败**"——
 * 这两类**必须分档**（混在一起会让真因被"权威离线"吃掉）。
 */
export const BASIC_ROUTES = ['/status', '/activity', '/history', '/config'] as const
