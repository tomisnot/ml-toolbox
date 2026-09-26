/**
 * 面板参数块：**本目录里唯一允许「项目专有」的文件**。
 *
 * ## 复制约定（见同目录 `README.md`）
 *
 * 把本目录复制进项目后：
 *   * **只许改这个文件**，且**只许改下面几个字段的值**；
 *   * **其余文件必须与 mecha 仓里的参考实现逐字一致** —— 判据比对的正是
 *     "除本文件之外"的正文（这样"漂移"就是机械可抓的，而不是靠人记得同步）。
 *
 * 为什么把它单独一个文件：**把可变量收进唯一一处**，比对才可能逐字。
 * 若允许"就地改几行"，逐字比对立刻失效，只能退化成人工 review。
 *
 * ## 三个字段
 *
 * * `ROUTE_PATH`：host 半注册的只读路由；client 半 fetch **同一个字符串**。
 *   两半同源同包 ⇒ "单一来源"在包内自动满足。**中性默认**（不含任何项目名），
 *   项目可覆盖。
 * * `PORT_FILE`：权威写的**端口文件**名（相对项目根）或绝对路径。
 *   ⚠ **故意留空、必须由项目填** —— 库**不替项目猜文件名**（同
 *   `resolve_data_dir(root, dir_name)` 的纪律：不给就报错）。留空时
 *   `resolveMonitorBase` 会**当场抛错**（响亮失败，不是静默用默认名）。
 * * `TITLE`：面板标题（纯展示；模型不读它）。
 *
 * ⚠ **本模块必须浏览器安全**（client 半会 import 它）⇒ 只许 import `./routes.ts`
 * 这类零 `node:*` 的模块。
 *
 * ## 本副本的两处合法差异（只有这个文件允许）
 * 1. 下面三个字段填的是 ML 的值；
 * 2. ⇒ 参考实现里那行 `import { DEFAULT_ROUTE_PATH } from './routes.ts'` 在本副本里
 *    **不再被用到，已按 R17 删除**（不留"声明了却不被读"的东西）。`routes.ts` 本身
 *    仍被 `monitor-client.ts` 使用（`BASIC_ROUTES`），所以它照样在闭包里。
 */

export interface PanelConfig {
  /** host 半注册的只读路由路径（client 半用同一字符串 fetch）。 */
  ROUTE_PATH: string
  /** 端口文件名（相对项目根）或绝对路径。**必填、无默认**。 */
  PORT_FILE: string
  /** 面板标题。 */
  TITLE: string
  /**
   * 标准页的**页签文案覆盖**（可选）。不填则用共享默认（`飞行记录仪` / `配置态`）。
   * ⚠ 只有标准两页能这样覆盖；**项目的附加页**文案写在它自己的**附加页声明**里。
   */
  TABS?: { rec?: string; cfg?: string }
}

export const PANEL_CONFIG: PanelConfig = {
  // ML 的值（本仓是唯一差异点；其余文件与 mecha 参考实现逐字一致）：
  // host 半注册它、client 半 fetch 它 —— 与 `dsh/src/index.ts` 的注册同源。
  ROUTE_PATH: '/ml-toolbox/monitor-url',
  // 权威（`ml_mecha.monitor_http` → `mecha.cockpit`）实际写的端口文件名。
  // ⚠ **单级、绝不回落**：这是相对上一版的**行为变更**（旧版先读
  // `.ml-mecha-runtime.json` 的 `monitor_port`，读不到再回落本文件）。
  PORT_FILE: '.ml-monitor-port',
  TITLE: 'ML 监控',
}
