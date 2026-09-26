/**
 * 面板**数据层**（无 react、无 DOM、无 node）：取址 → 取数 → 归约成**唯一状态** → 文案/徽标。
 *
 * ## 为什么单独一层
 *
 * 1. **判据要落在消费者这一层**：这一层是纯的，Node 能直接喂真形状/坏形状
 *    （react 是 dsh 宿主提供的 peer，本仓 devDeps 里没有 ⇒ **渲染层不进判据**）；
 * 2. **"面板不空白"要成为可测的性质**：状态 → 文案/徽标的映射是纯函数，判据能**逐档**
 *    断言"有可读文本"；`.tsx` 只负责把它塞进 DOM。
 *
 * ## 取数全走同目录的共享模块
 *
 * `fetchMonitorBase` / `readJson` / `panelState` / `displayTier` / `statsLine` 来自
 * `./monitor-client.ts`：
 * * **三档错误态**（`address` 拿不到地址 / `offline` 基础路由失败 / `route` 附加路由失败）
 *   **分开**——混在一起会让真因被"权威离线"吃掉；
 * * **`empty` 档**：连上了、也 200、但一点数据都没有 ⇒ 单列一档
 *   （没有它，"取数成功但面板空白"会伪装成成功）；
 * * **零回落**：地址拿不到就是拿不到，绝不返回空地址/相对路径去 fetch。
 */
import {
  displayTier,
  fetchMonitorBase,
  isNonDegenerate,
  panelState,
  readJson,
  statsLine,
  type FetchResult,
  type PanelState,
} from './monitor-client.ts'
import { PANEL_CONFIG } from './panel-config.ts'
// ⚠ 这两个只在**类型位置**用（`eventsOf` / `configOf` 的返回类型）⇒ 运行时被剥掉，
// 但**类型检查需要它们被 import**：漏了会报 `TS2304: Cannot find name 'Ev'`，
// 而**框架自己的 `node --test` 看不见**（类型剥离不做检查）。
// ⇒ 这正是"资产发布了框架无法类型检查的代码、第一个消费者付账"那个缺口；
// 现在由 `checks/dsh_panel_selfcheck.py` 的**类型检查守卫**兜住（缺 tsc 即红）。
import type { ConfigWire, Ev } from './panel-view.ts'

/**
 * 路由与请求路径。
 *
 * ⚠ **键必须用 `BASIC_ROUTES` 的原样成员**（`/history`、`/config`）：`panelState` 是拿
 * **键**去索引 payloads 再喂 `monitorStats` 的（`payloads['/history']`）。若把键写成带查询串的
 * `/history?since_seq=0`，事件数会**恒为 0**（真实踩过：自证行打出"事件行=0"而数据其实到了）
 * ——查询串只属于**请求路径**，不属于键。
 */
export const HISTORY_ROUTE = '/history'
export const CONFIG_ROUTE = '/config'
/** 实际请求路径：全量尾（`since_seq=0`），键仍用 `HISTORY_ROUTE`。 */
export const HISTORY_PATH = '/history?since_seq=0'
export const CONFIG_PATH = CONFIG_ROUTE

/** 标准页的 id（项目的附加页用它们自己的 id，见 `ExtraPageSpec`）。 */
export type StandardPage = 'rec' | 'cfg'

/** 页签文案：**共享默认 + 项目可覆盖**（覆盖在参数块 `TABS` 里）。 */
export function tabLabel(id: string): string {
  const over = PANEL_CONFIG.TABS ?? {}
  if (id === 'rec') return over.rec ?? '飞行记录仪'
  if (id === 'cfg') return over.cfg ?? '配置态'
  return id
}

/**
 * 一拍取数：地址 → 两条基础路由 → 归约（+ 可选的**附加页**取数）。
 *
 * ⚠ **地址拿不到时不再 fetch 任何东西**：不拿空 base 去拼相对路径，否则
 * "软件没在 AI 模式跑"会被显示成"连上了但面板空白"。
 *
 * ⚠ **附加页的失败不牵连标准两页**：它们各自的 `FetchResult` 单独返回，由各自那一页
 * 渲染可读错误（`unreadableText`）。若把它们塞进 `panelState` 的 `extra`，一处附加路由
 * 失败会把**整个面板**打成 `route` 档——那会把"记录仪/配置态本来是好的"也一起吃掉。
 */
export interface ExtraRoute { id: string; path: string }

export interface PanelLoad {
  state: PanelState
  /** 附加页各自的取数结果（按 `id`）。没有声明附加页时是空对象。 */
  extras: Record<string, FetchResult<unknown>>
}

export async function loadPanel(doFetch: typeof fetch = fetch,
                                extraRoutes: ExtraRoute[] = []): Promise<PanelLoad> {
  const base = await fetchMonitorBase(doFetch, PANEL_CONFIG.ROUTE_PATH)
  if (base.ok !== true) return { state: panelState({ base }), extras: {} }
  const history = await readJson(base.data, HISTORY_PATH, doFetch)
  const config = await readJson(base.data, CONFIG_PATH, doFetch)
  const state = panelState({ base, basic: { [HISTORY_ROUTE]: history, [CONFIG_ROUTE]: config } })
  const extras: Record<string, FetchResult<unknown>> = {}
  for (const r of extraRoutes) {
    extras[r.id] = await readJson(base.data, r.path, doFetch)
  }
  return { state, extras }
}

/** 状态 → **可读文案**（`ok` 档返回空串：那一档要画数据，不画文案）。 */
export function stateText(state: PanelState): string {
  switch (state.kind) {
    case 'ok':
      return ''
    case 'address':
    case 'offline':
      // 显示档位把 address 显式塌缩成 offline（共享资产的 `displayTier`）：两者对人
      // 是同一件事——"权威没起来"；但**档位本身**仍分开（日志/判据里能辨真因）。
      return `连不上权威（软件没在 AI 模式跑？）：${state.detail}`
    case 'route':
      return `权威在，但附加路由失败：${state.detail}`
    case 'empty':
      return `连上了但没拿到数据（${state.detail}）`
    default:
      return assertNeverText(state)
  }
}

/** 状态 → 徽标（类名 + 文案）。`null` = 首拍还没回来（**不是空白**）。 */
export function stateBadge(state: PanelState | null): { cls: string; text: string } {
  if (!state) return { cls: 'ok', text: '连接中…' }
  switch (state.kind) {
    case 'ok': {
      const mode = modeOf(state)
      return {
        cls: mode === 'ai' ? 'ai' : mode === 'locked' ? 'warn' : 'ok',
        text: mode === 'ai' ? '写权 AI' : mode === 'locked' ? '全员禁写'
          : mode === 'human' ? '写权在人' : '连接中…',
      }
    }
    case 'address':
    case 'offline':
      return { cls: 'error', text: displayTier(state.kind) === 'offline' ? '权威离线' : '地址未知' }
    case 'route':
      return { cls: 'warn', text: '路由异常' }
    case 'empty':
      return { cls: 'warn', text: '连上但无数据' }
    default:
      return assertNeverBadge(state)
  }
}

/** `ok` 档里取写权模式（面板拿 `mode` 做徽标）。 */
export function modeOf(state: PanelState): string {
  if (state.kind !== 'ok') return ''
  const h = state.payloads[HISTORY_ROUTE] as { mode?: unknown } | undefined
  return typeof h?.mode === 'string' ? h.mode : ''
}

/** R8 自证行（**判据/日志必须把它打出来**，否则"非退化"只是声称）。 */
export function selfCheckLine(state: PanelState): string {
  if (state.kind !== 'ok') return stateText(state)
  return statsLine(state.stats)
}

/** `ok` 档是否非退化（`empty` 档就是它的反面）。 */
export function okIsNonDegenerate(state: PanelState): boolean {
  return state.kind === 'ok' && isNonDegenerate(state.stats)
}

/**
 * 从状态里取**写事件**（`ok` 档才有；其余档给空数组 + 由文案负责解释）。
 *
 * ⚠ **键必须是 `HISTORY_ROUTE`（不带查询串）**：`panelState` 的 `payloads` 是按
 * **基础路由键**索引的。若这里写成请求路径 `HISTORY_PATH`（带 `?since_seq=0`），
 * 取到的是 `undefined` ⇒ **页面永远显示"暂无写事件"**，而**自证行仍报"事件行=N"**
 * ——两个地方用了两个键，页面对不上自证。抽成这个被单测覆盖的纯函数，就是为了
 * 让这个键只有一处、且被钉住。
 */
export function eventsOf(state: PanelState | null): Ev[] {
  if (!state || state.kind !== 'ok') return []
  const h = state.payloads[HISTORY_ROUTE] as { events?: Ev[] } | undefined
  return Array.isArray(h?.events) ? h!.events! : []
}

/** 从状态里取 `/config` 的 wire 形状（`ok` 档才有）。键同样用 `CONFIG_ROUTE`。 */
export function configOf(state: PanelState | null): ConfigWire | undefined {
  if (!state || state.kind !== 'ok') return undefined
  return state.payloads[CONFIG_ROUTE] as ConfigWire | undefined
}

/** 兜底：`switch` 到不了的分支**抛**，绝不"什么都不画"。 */
function assertNeverText(x: never): never {
  throw new Error(`未处理的 PanelState（文案）：${JSON.stringify(x)}`)
}

function assertNeverBadge(x: never): never {
  throw new Error(`未处理的 PanelState（徽标）：${JSON.stringify(x)}`)
}
