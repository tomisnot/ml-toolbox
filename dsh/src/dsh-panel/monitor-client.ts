/**
 * 面板**取数骨架**（client 半）：同源取址 + 四基础路由取数 + **错误档位** +
 * **R8 非退化自证** + **把"空白"变成不可表达的状态**。**参考实现**，不是框架能力。
 *
 * ## 抽什么、不抽什么
 *
 * * **抽**：`fetchMonitorBase`（同源 fetch 地址路由）/ `readJson` 统一取数 /
 *   **错误档位**（`address` `offline` `route` `empty`）/ `panelState` 归约 /
 *   非退化自证计数。
 * * **不抽**：任何**领域页**（谁在附加路由上画什么）与**文案/渲染**——那是项目的。
 *
 * ## 错误档位（**档位共享，文案归项目**）
 *
 * | 档 | 含义 |
 * |---|---|
 * | `address` | **地址拿不到**：路由 503/连不上/正文不是 JSON/`base` 形状不对 |
 * | `offline` | **权威离线**：**基础路由**（`BASIC_ROUTES`）失败 |
 * | `route` | **附加路由失败**：权威在，但某条项目自己的附加路由失败 |
 * | `empty` | ⭐ **连上了、也 200，但一点数据都没有**（R8：非退化不成立） |
 *
 * ⚠ `empty` 这一档是**故意存在**的：没有它，"取数成功但面板空白"就会伪装成成功——
 * 那正是"两边都空所以相同"的假绿。
 *
 * ⚠ 把附加路由的失败混进"权威离线"会让**真因被吃掉**（本工程最贵的一类假象）。
 *
 * ## ⚠ 本模块必须**浏览器安全**（零 `node:*`）
 *
 * 它是 **client 半**。**不许** import `monitor-url.ts`（那个是 node-only 的 host 半：
 * 顶层 `node:fs`）。共享常量住 `routes.ts` —— 2026-09-26 的真实缺陷就是这里原先
 * 从 `monitor-url.ts` 取值导入 `BASIC_ROUTES`，把 `node:fs` 拖进了原生消费者的
 * 浏览器 bundle（加载即失败）。**边界不靠打包器树摇的运气**，靠
 * `monitor-client.test.ts` 的**导入闭包检查**。
 */
import { BASIC_ROUTES, DEFAULT_ROUTE_PATH } from './routes.ts'

export type MonitorFailureTier = 'address' | 'offline' | 'route'

/** 失败事实：**档位是机制、文案是项目**。 */
export interface MonitorFailure {
  tier: MonitorFailureTier
  detail: string
}

/** 取数结果：要么拿到数据，要么拿到一个**带档位**的失败。 */
export type FetchResult<T> = { ok: true; data: T } | { ok: false; failure: MonitorFailure }

export type DoFetch = typeof fetch

/** 该路径属于"基础路由"还是"附加路由"（决定失败落到哪一档）。 */
export function classifyRoute(path: string): 'basic' | 'extra' {
  // ⚠ `?? ''`：严格 tsconfig（`noUncheckedIndexedAccess`）下 `split(...)[0]` 是
  // `string | undefined` ⇒ 会报 TS2345。**这是可移植性要求的一部分**（见 README）：
  // 资产必须能在**最严的**消费者 tsconfig 下编译，不能让项目为它放松类型严格度。
  const clean = path.split('?')[0] ?? ''
  return (BASIC_ROUTES as readonly string[]).includes(clean) ? 'basic' : 'extra'
}

/** 地址形状自检：只接受 `http(s)://…` 这种**像源**的字符串。 */
export function isOriginLike(base: unknown): boolean {
  return typeof base === 'string' && /^https?:\/\/\S+$/.test(base.trim())
}

/**
 * 取监控端点基址：**同源** `fetch(routePath)` → `{base}`。
 *
 * * `doFetch` 是**真注入的缝**（判据可在 Node 里喂真形状与各种失败形状）；
 * * 失败一律落 `address` 档，**绝不返回空地址冒充成功**；
 * * ⚠ 形状不对（空 / 不是 `http(s)://…`）也归 `address` 档：否则"垃圾地址"会一路
 *   走到 connect 失败、被显示成"权威离线"，**真因被吃掉**。
 */
export async function fetchMonitorBase(
  doFetch: DoFetch = fetch,
  routePath: string = DEFAULT_ROUTE_PATH,
): Promise<FetchResult<string>> {
  let res: Response
  try {
    res = await doFetch(routePath, { cache: 'no-store' })
  } catch (err) {
    return { ok: false, failure: { tier: 'address', detail: `地址路由连不上：${(err as Error).message}` } }
  }
  if (!res.ok) {
    // 地址路由自己回 503（权威没起来时它就是这么答的）⇒ 仍是"地址拿不到"
    return { ok: false, failure: { tier: 'address', detail: `地址路由返回 ${res.status}` } }
  }
  let payload: unknown
  try {
    payload = await res.json()
  } catch (err) {
    return { ok: false, failure: { tier: 'address', detail: `地址路由正文不是 JSON：${(err as Error).message}` } }
  }
  const base = (payload as { base?: unknown } | null)?.base
  if (!isOriginLike(base)) {
    return { ok: false, failure: { tier: 'address', detail: `地址形状不对：${JSON.stringify(base)}` } }
  }
  return { ok: true, data: (base as string).trim() }
}

/**
 * 统一取数：`GET {base}{path}` → JSON。
 *
 * * 非 2xx / 连不上 / 解析失败 ⇒ 失败，档位由 `classifyRoute(path)` 决定；
 * * **不做任何回落**：失败就是失败，**不返回空对象冒充成功**。
 */
export async function readJson(
  base: string,
  path: string,
  doFetch: DoFetch = fetch,
): Promise<FetchResult<unknown>> {
  const tier: MonitorFailureTier = classifyRoute(path) === 'basic' ? 'offline' : 'route'
  let res: Response
  try {
    res = await doFetch(`${base}${path}`, { cache: 'no-store' })
  } catch (err) {
    return { ok: false, failure: { tier, detail: `${path} 连不上：${(err as Error).message}` } }
  }
  if (!res.ok) {
    return { ok: false, failure: { tier, detail: `${path} 返回 ${res.status}` } }
  }
  try {
    return { ok: true, data: await res.json() }
  } catch (err) {
    return { ok: false, failure: { tier, detail: `${path} 正文不是 JSON：${(err as Error).message}` } }
  }
}

/** 显示档位：把 `address` **显式**塌缩成 `offline`（"权威没起来"是实情）。 */
export function displayTier(tier: MonitorFailureTier): MonitorFailureTier {
  return tier === 'address' ? 'offline' : tier
}

// ---------------------------------------------------------------- R8 自证

export interface HistoryPayload { events?: unknown[] }
export interface ConfigPayload { groups?: Record<string, unknown> }

export interface MonitorStats {
  offline: boolean
  events: number
  configFamilies: number
}

/**
 * 非退化自证计数（**R8**）：判"面板正常"的守卫必须**同时**证明"数据真的到了"——
 * 否则"空表 vs 空表""离线 vs 离线"同样会通过，绿得毫无信息。
 */
export function monitorStats(input: {
  offline?: boolean
  history?: HistoryPayload
  config?: ConfigPayload
}): MonitorStats {
  const events = Array.isArray(input.history?.events) ? input.history!.events!.length : 0
  const groups = input.config?.groups
  const configFamilies = groups && typeof groups === 'object' ? Object.keys(groups).length : 0
  return { offline: Boolean(input.offline), events, configFamilies }
}

/** 自证行（判据**必须把它打出来**，否则"非退化"只是声称）。 */
export function statsLine(stats: MonitorStats): string {
  return `离线=${stats.offline} 事件行=${stats.events} 配置族=${stats.configFamilies}`
}

/** 非退化判定：**不在离线态**，且**至少拿到了一样东西**。 */
export function isNonDegenerate(stats: MonitorStats): boolean {
  return !stats.offline && (stats.events > 0 || stats.configFamilies > 0)
}

// ---------------------------------------------------------------- 归约

/**
 * 面板状态：**判别联合**——渲染方必须**逐个分支**处理它。
 *
 * ⭐ 为什么要有它：**让"面板空白"在类型上不可表达**。没有它，项目完全可以写出
 * "取数失败就什么也不画"的分支（失败与成功都渲染成空气），而那正是"静默空白"
 * 那一族缺陷的入口。有了它，**每个分支都带 detail 或数据**，没有"什么都不画"的档。
 */
export type PanelState =
  | { kind: 'ok'; stats: MonitorStats; payloads: Record<string, unknown> }
  | { kind: 'address'; detail: string }
  | { kind: 'offline'; detail: string; failed: string[] }
  | { kind: 'route'; detail: string; failed: string[] }
  | { kind: 'empty'; detail: string }

/**
 * 把各处取数结果归约成**唯一一个**面板状态（**纯函数**，不做任何 I/O）。
 *
 * 判定顺序（前面的赢）：地址拿不到 → 基础路由失败 → 附加路由失败 → 非退化不成立 → ok。
 */
export function panelState(input: {
  base?: FetchResult<string>
  basic?: Record<string, FetchResult<unknown>>
  extra?: Record<string, FetchResult<unknown>>
}): PanelState {
  const base = input.base
  if (!base || base.ok !== true) {
    return {
      kind: 'address',
      detail: base && base.ok === false ? base.failure.detail : '没有取址结果（面板还没拿到地址）',
    }
  }
  const basic = input.basic ?? {}
  const basicFailed = Object.entries(basic)
    .filter(([, r]) => r.ok !== true)
    .map(([p]) => p)
  if (basicFailed.length) {
    return {
      kind: 'offline',
      detail: `${basicFailed.join('、')} 失败（权威离线？）`,
      failed: basicFailed,
    }
  }
  const extra = input.extra ?? {}
  const extraFailed = Object.entries(extra)
    .filter(([, r]) => r.ok !== true)
    .map(([p]) => p)
  if (extraFailed.length) {
    return { kind: 'route', detail: `${extraFailed.join('、')} 失败（权威在，是这些路由的问题）`,
             failed: extraFailed }
  }
  const payloads: Record<string, unknown> = {}
  for (const [p, r] of Object.entries(basic)) {
    if (r.ok === true) payloads[p] = r.data
  }
  const stats = monitorStats({
    offline: false,
    history: payloads['/history'] as HistoryPayload | undefined,
    config: payloads['/config'] as ConfigPayload | undefined,
  })
  if (!isNonDegenerate(stats)) {
    // ⭐ 连上了、也 200，但一点数据都没有：**不许**当成成功（否则就是空白面板的假绿）
    return { kind: 'empty', detail: `连上了但没拿到数据（${statsLine(stats)}）` }
  }
  return { kind: 'ok', stats, payloads }
}

/**
 * 渲染方的**兜底**：`switch` 到不了的分支交给它。
 *
 * 类型侧：`x: never` ⇒ 有未处理的 `kind` 就**编译不过**（若项目跑了类型检查）。
 * 运行侧：真收到未处理的状态就**抛**——绝不"什么都不画"。
 * ⚠ 本仓的自测**只跑类型剥离、不做类型检查**（见 README「未验证」）⇒ 我们只实测了
 * 运行侧这条兜底；类型侧那条**在跑了类型检查的项目里**才成立。
 */
export function assertNever(x: never): never {
  throw new Error(`未处理的 PanelState：${JSON.stringify(x)}`)
}
