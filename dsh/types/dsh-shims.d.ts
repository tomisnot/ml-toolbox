/**
 * 本地**最小类型 shim** —— 只声明本插件实际用到的 dsh / Cordis API 表面。
 *
 * ## 为什么是 shim 而不是装真包
 *
 * `@deepseek-ai/cordis` 等是 **dsh 宿主运行时提供**的 peer（未发布 npm），out-of-tree
 * 插件本地装不到，也**不该**把某台机器上 dsh 检出路径的绝对地址写进交付 tsconfig。
 * shim 只声明"我们实际用到的那一小片"，形状取自本插件的真实用法；运行时由 dsh 提供
 * 真实现。⚠ **保持最小**：用到新的成员再补（EL 的同名文件是 158 行，因为它还带 MCP 桥 /
 * 命令 / slots 全家桶；本插件只做"地址路由 + 面板按钮"，所以这里短得多）。
 *
 * ## 这份 shim 的消费者（R1：写明谁在用）
 *
 * * `src/index.ts`（host 半）：`Context.effect` / `Context.webServer.register`；
 * * `src/client/index.ts`（client 半）：`Context.effect` / `slots` / `layout` /
 *   `sidebarRight` / `sidebarRightTabs`；
 * * `tsconfig.json` 的 `include` 收了 `types` 目录下的 `.d.ts`，由 `npm run typecheck` 消费。
 */

declare module '@deepseek-ai/cordis' {
  /** 一个 slot 上的注册项（本插件只用到这几个字段）。 */
  export interface SlotRegistration {
    name: string
    id?: string
    key?: string
    order?: number
    label?: string
    inject?: (...args: unknown[]) => unknown
  }

  export interface SlotsRuntime {
    /** 往某个 slot 注入/注册一个组件，返回 disposer。 */
    inject(name: string, fn: () => unknown): () => void
    register(spec: SlotRegistration, component: unknown): () => void
  }

  export interface LayoutRuntime {
    /** 开/关右栏。 */
    openRightbar?(open: boolean, focus?: boolean): void
  }

  export interface SidebarRightRuntime {
    /** 切到某个 tab。 */
    openTab?(id: string): void
  }

  export interface SidebarRightTabsRuntime {
    /** 注册一个右栏 tab 类型，返回可选的释放函数。 */
    register?(spec: { id: string; kind: string; title: () => string }):
      (() => void) | undefined
  }

  export interface WebServerRuntime {
    /** 注册一条只读路由（`kind: 'exact'` + 绝对路径），返回 disposer。 */
    register(spec: {
      kind: 'exact'
      path: string
      handler: (req: unknown, res: unknown) => void
    }): () => void
  }

  export interface Logger {
    info(...args: unknown[]): void
    warn(...args: unknown[]): void
    error(...args: unknown[]): void
  }

  /** Cordis 上下文（服务容器）：只声明本插件用到的成员。 */
  export interface Context {
    readonly webServer: WebServerRuntime
    readonly slots: SlotsRuntime
    readonly layout?: LayoutRuntime
    readonly sidebarRight?: SidebarRightRuntime
    readonly sidebarRightTabs?: SidebarRightTabsRuntime
    readonly logger?: Logger
    /** 把一段副作用绑到插件 fiber 上；回调返回 disposer。 */
    effect(fn: () => (() => void) | void, label?: string): Promise<unknown>
  }
}
