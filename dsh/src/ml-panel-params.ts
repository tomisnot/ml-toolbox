/**
 * ML 的面板**参数家**（依赖化之后 `panel-config.ts` 副本已删 ⇒ 参数必须住进项目自己）。
 *
 * ## 为什么必须有这个文件
 *
 * 依赖化之前，ML 的这三个值住在**副本** `src/dsh-panel/panel-config.ts` 里（"复制一份、
 * 只改参数块"）。改成 `@mecha/dsh-panel` 的 `file:` 依赖之后，三个项目经 junction 指向
 * **同一份**资产 ⇒ 谁也不能再改资产里的那个文件（互相覆盖）。所以参数**搬进项目**，
 * 由项目启动时**注入**一次。
 *
 * ⚠ 这也**替掉了旧指纹判据里那两条"参数块确实填了 ML 的值"的断言**
 * （原先它靠正则匹 `ROUTE_PATH: '/ml-toolbox/monitor-url'` 钉字面量）。
 * 新家由两件事守：① 本文件就是路由字面量的**唯一**住处（闭包/单一来源判据扫它）；
 * ② `test/ml-panel.test.ts` 断言"注入后能原样读回这三个值"。
 *
 * ## 形状必须是小写三字段
 *
 * 资产 `PanelOptions` 是 **`{ routePath, portFile, title, tabs? }`**（三个必填、**无默认**，
 * 资产不替项目猜名字）—— 与旧副本的**大写** `PANEL_CONFIG = { ROUTE_PATH, PORT_FILE, TITLE }`
 * 形状不同，不是改名。`tabs?` 本仓不用（用共享默认页签文案）。
 *
 * ## 谁来注入（两半各一次，不能只注一处）
 *
 * `configurePanel(...)` 是**显式**注入，且 `panelConfig()` **未注入即抛**。host 半与 client 半
 * 是**两份 bundle、两个进程**（client 在浏览器里跑）⇒ 各注入各的：
 * - host 半：`src/index.ts` 模块顶层注入；
 * - client 半：`src/client/index.ts` 模块顶层注入。
 *
 * ⚠ 类型检查**不会**替你发现漏注入（漏了照样过 `tsc`，到运行时才炸）——所以有一条
 * 不依赖浏览器的判据专门验它（`test/ml-panel.test.ts` 的"注入往返"那条）。
 */
import type { PanelOptions } from '@mecha/dsh-panel/panel/panel-config.ts'

/**
 * ML 注入给共享面板的三个值（**唯一来源**；路由字面量只许出现在本文件）。
 *
 * - `routePath`：host 半注册的只读路由；client 半 fetch **同一个字符串**。
 *   与 `src/index.ts` 早先导出的 `ROUTE_PATH` 同值（该导出已删除：它是第二处字面量的温床）。
 * - `portFile`：**权威实际写**的那个端口文件名（`ml_mecha/runtime.py` 的 `MONITOR_PORT_FILE`）。
 *   ⚠ 它必须与 Python 侧一致；写错的表现是**一律 503**（资产刻意如此，不回落）。
 * - `title`：右栏页签标题（纯展示，模型不读它）。
 */
export const ML_PANEL: PanelOptions = {
  routePath: '/ml-toolbox/monitor-url',
  portFile: '.ml-monitor-port',
  title: 'ML 监控',
}
