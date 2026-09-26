/**
 * 面板**视图层**：**纯字符串生成**（无 react、无 DOM、无 node）⇒ 可被 `node:test` 直接测。
 *
 * ## 为什么是字符串而不是 JSX
 *
 * 面板的呈现本来就是"数据 → HTML 文本"的映射：**布局是字符串，React 只在外层壳上**
 * （挂载、轮询、事件委托，见同目录 `MonitorTabBody.tsx`）。把这一层做成纯函数，
 * **"面板不空白"就变成一条字符串级、可逐档断言的性质**——那是本工程最想钉住的东西。
 *
 * ## 两条纪律
 *
 * 1. **展开态由调用方给**（`isOpen` 谓词），**不读模块级集合**：隐藏状态会让判据之间
 *    互相污染、且"换掉它结果就变"测不出来（R17）。壳里那份模块级集合只负责**记住**，
 *    渲染只问谓词。
 * 2. **类名中性**（`mecha-` 前缀）：本目录是**共享资产**，不许出现任何项目前缀或项目字面量
 *    （词表守卫在扫 `.ts`/`.tsx`/`.md`）。
 */
import type { FetchResult } from './monitor-client.ts'

/** 一条写事件（框架 History 的 wire 形状）。 */
export interface Ev {
  seq: number
  kind: string
  actor: string
  target: string
  before: unknown
  after: unknown
  reason?: string
  /** 与调用方的行为史互引（可选：老事件没有这个字段）。 */
  call_id?: string
}

/** 配置态里的一行（框架 `config_tree` 的 wire 形状）。 */
export interface Row {
  name: string
  short: string
  value: unknown
  unit?: string
  set?: { actor: string; seq: number } | null
}

/** 配置态里的一个前缀分组。 */
export interface Sub { prefix: string; rows: Row[] }

/** `/config` 的 wire 形状（框架契约）。 */
export interface ConfigWire {
  groups?: Record<string, Sub[]>
  has_schema?: boolean
  orphans?: Row[]
}

/** 展开态谓词：`true` = 该 key 展开。**由调用方给**（判据可喂任意实现）。 */
export type IsOpen = (key: string) => boolean

/** HTML 转义（本层所有插入点都必须过它）。 */
export const esc = (x: unknown): string =>
  String(x ?? '').replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string))

/**
 * 值的**友好显示**：不 dump 转义 JSON（防"乱码"）。
 *
 * - 空值 ⇒ `∅`
 * - 指针（`$pointer`）⇒ `⟨数组 a×b⟩`
 * - 截断（`$truncated`）⇒ `⟨截断 kept/total⟩`
 * - 其它对象 ⇒ JSON，超过 120 字符截断加 `…`
 */
export const fmtVal = (v: unknown): string => {
  if (v === null || v === undefined) return '∅'
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>
    if (o.$pointer) return '⟨数组 ' + ((o.shape || []) as unknown[]).join('×') + '⟩'
    if (o.$truncated) return '⟨截断 ' + o.kept + '/' + o.total + '⟩'
    const s = JSON.stringify(v)
    return s && s.length > 120 ? s.slice(0, 117) + '…' : s ?? String(v)
  }
  return String(v)
}

/** 页签条（`active` 的那个加 `on`）。`badgeHtml` 由调用方给（它带档位语义）。 */
export function tabsHtml(pages: { id: string; label: string }[], active: string,
                         badgeHtml: string): string {
  return '<div class="mecha-tabs">' + pages.map((p) =>
    `<button data-page="${esc(p.id)}" class="${p.id === active ? 'on' : ''}">` +
    `${esc(p.label)}</button>`).join('') + badgeHtml + '</div>'
}

/** 状态徽标。 */
export function badgeHtml(cls: string, text: string): string {
  return `<span class="mecha-badge ${esc(cls)}">${esc(text)}</span>`
}

/**
 * 飞行记录仪页（**逐行折叠**：折叠=概括 seq/actor/kind/reason，展开=before→after 全量）。
 *
 * 空表时**返回可读文案**（不是空串）——"面板不空白"在这一层就要成立。
 */
export function recPageHtml(evs: Ev[], isOpen: IsOpen): string {
  if (!evs.length) {
    return '<div class="mecha-msg muted">暂无写事件——AI/人每改一个键都会出现在这里</div>'
  }
  const rows = evs.slice(-60).reverse().map((e) => {
    const k = 'ev:' + e.seq
    const open = isOpen(k) ? ' open' : ''
    return `<details class="mecha-ev" data-key="${esc(k)}"${open}><summary>` +
      `<span class="seq">#${esc(e.seq)}</span>` +
      `<span class="${e.actor === 'ai' ? 'actor-ai' : ''}">${esc(e.actor)}</span>` +
      `<span class="mecha-tag">${esc(e.kind)}</span>` +
      `<span class="mecha-reason" title="${esc(e.target)}">${esc(e.reason || e.target)}</span>` +
      `</summary><div class="body">` +
      `<div><i>target</i>${esc(e.target)}</div>` +
      `<div><i>before → after</i>${esc(fmtVal(e.before))} → ${esc(fmtVal(e.after))}</div>` +
      `<div><i>reason</i>${esc(e.reason || '')}</div>` +
      `<div><i>call_id</i>${esc(e.call_id || '')}</div>` +
      `</div></details>`
  }).join('')
  return `<div class="mecha-page">${rows}</div>`
}

/**
 * 配置态页：**族 → 前缀**二级折叠树（显示名去前缀）+ **契约外（orphans）**表。
 *
 * `has_schema` 为假时**如实说明**（"无 schema——下方按快照直接列出现值"），
 * 而不是画一棵空树。
 */
export function cfgPageHtml(cfg: ConfigWire | undefined, isOpen: IsOpen): string {
  const groups = cfg?.groups ?? {}
  const orphans = cfg?.orphans ?? []
  const tree = Object.entries(groups).map(([fam, subs]) =>
    `<details class="mecha-fam" data-key="fam:${esc(fam)}"` +
    `${isOpen('fam:' + fam) ? ' open' : ''}>` +
    `<summary>${esc(fam)}</summary>` +
    (subs || []).map((sg) =>
      `<details class="mecha-sub" data-key="sub:${esc(fam)}/${esc(sg.prefix)}"` +
      `${isOpen('sub:' + fam + '/' + sg.prefix) ? ' open' : ''}>` +
      `<summary>${esc(sg.prefix || '(top)')} (${(sg.rows || []).length})</summary><table>` +
      (sg.rows || []).map((r) =>
        `<tr><td title="${esc(r.name)}${r.set ? ' ← ' + esc(r.set.actor) + ' #' + esc(r.set.seq) : ''}">` +
        `${esc(r.short || r.name)}</td>` +
        `<td>${esc(fmtVal(r.value))}${r.unit ? ' <i>' + esc(r.unit) + '</i>' : ''}</td></tr>`
      ).join('') +
      '</table></details>').join('') +
    '</details>').join('')
  const noSchema = cfg?.has_schema ? '' :
    '<div class="mecha-msg muted">无 schema——下方按当前快照直接列出现值</div>'
  const orphanTable = orphans.length
    ? `<h4>契约外（orphans）</h4><table>${orphans.map((r) =>
        `<tr><td title="${esc(r.name)}">${esc(r.name)}</td>` +
        `<td>${esc(fmtVal(r.value))}</td>` +
        `<td class="muted">${r.set ? esc(r.set.actor) + ' #' + esc(r.set.seq) : ''}</td></tr>`
      ).join('')}</table>`
    : ''
  return `<div class="mecha-page"><h4>配置态</h4>${noSchema}${tree}${orphanTable}</div>`
}

// ---------------------------------------------------------------- 项目附加页（声明式）

/**
 * 附加页的**一列**：`cell` 由**项目**给（取值 + 格式化）⇒ 领域词留在项目里。
 */
export interface ExtraColumn {
  label: string
  cell: (row: unknown) => string
}

/**
 * **项目的附加页声明**（声明式列表，**不是插件/hook 框架**）。
 *
 * ## 它的存在理由（不是"为将来"）
 *
 * 并入某个项目的面板时，它那份"运行记录"页含**该项目自己的领域词**（方法名、指标名…）。
 * 原样搬进资产 ⇒ **领域词静默进了框架**（L5），而**词表拦不住**（词表是另一套领域值）。
 * ⇒ **声明式**是唯一能让"能力并入、领域词不进"的结构性答案。
 *
 * ## 数据契约（**共享面板定 wire 形状**）
 *
 * 附加路由**必须**回 `{ rows: [...], ... }`：表格行取 `payload.rows`；其余键由
 * `summaryLine` 自己解释（它拿到整个 payload）。⇒ 这样 spec 不需要"行提取器"字段。
 *
 * ## 字段冻结在 **8 个**
 *
 * 再加字段要**先有第二个页面型消费者**（今天只有一家）——否则就是在为一个消费者把 spec 撑大。
 */
export interface ExtraPageSpec {
  /** 页签标识（项目词，只在本项目内唯一）。 */
  id: string
  /** 页签文案（项目词）。 */
  label: string
  /** 附加路由的**请求路径**（含查询串，如 `/runs?limit=50`）。**资产里不许出现这个字面量。** */
  path: string
  /** 页内标题（项目词）。 */
  title: string
  /** 表格列（项目给 cell）。 */
  columns: ExtraColumn[]
  /** 空表文案（项目词）。⚠ **必须与"读不到"分开**。 */
  emptyText: string
  /** 读不到时的文案（项目词）。**面板必画它**，不许渲染成空表。 */
  unreadableText: string
  /** 顶部概括**一行**（可选；不声明就不画）。项目自己的词与拼法都在这个函数里。 */
  summaryLine?: (payload: unknown) => string
  /** 常驻脚注（可选；如"存档为空不等于没跑过"）。 */
  note?: string
}

/** 附加页：**这一页自己的**失败只影响这一页（不牵连标准两页）。 */
export function extraPageHtml(spec: ExtraPageSpec, result: FetchResult<unknown>): string {
  const head = `<h4>${esc(spec.title)}</h4>`
  const note = spec.note ? `<div class="mecha-hint">${esc(spec.note)}</div>` : ''
  if (result.ok !== true) {
    // ⭐ **读不到 ≠ 空表**：把真因摆成人能读的文本，并明说"这一格不是空的"。
    return `<div class="mecha-page">${head}` +
      `<div class="mecha-msg">${esc(spec.unreadableText)}：${esc(result.failure.detail)}</div>` +
      note + '</div>'
  }
  const payload = result.data as { rows?: unknown[] } | null
  const rows = Array.isArray(payload?.rows) ? payload!.rows! : []
  const summary = spec.summaryLine ? spec.summaryLine(payload) : ''
  const headLine = summary
    ? `<div class="mecha-headline">${esc(summary)}</div>`
    : ''
  if (!rows.length) {
    return `<div class="mecha-page">${head}${headLine}` +
      `<div class="mecha-hint">${esc(spec.emptyText)}</div>${note}</div>`
  }
  const table = '<table>' + rows.map((r) =>
    '<tr>' + spec.columns.map((c, i) =>
      `<td class="${i ? '' : 'key'}">${esc(c.cell(r))}</td>`).join('') + '</tr>').join('') +
    '</table>'
  const header = '<table><tr>' + spec.columns.map((c) =>
    `<td class="muted">${esc(c.label)}</td>`).join('') + '</tr></table>'
  return `<div class="mecha-page">${head}${headLine}${header}${table}${note}</div>`
}

/** 面板样式（**布局本体**；中性前缀，随资产发货）。 */
export const TAB_CSS = `
.mecha-tab-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.mecha-tab-native .mecha-tabs{position:sticky;top:0;display:flex;gap:4px;align-items:center;padding:6px 8px;background:#f6f8fa;border-bottom:1px solid #d0d7de;z-index:2}
.mecha-tab-native .mecha-tabs button{cursor:pointer;border:1px solid #d0d7de;background:#fff;border-radius:6px;padding:3px 10px;font:inherit;color:#57606a}
.mecha-tab-native .mecha-tabs button.on{background:#0969da;border-color:#0969da;color:#fff}
.mecha-tab-native .mecha-badge{margin-left:auto;padding:2px 8px;border-radius:10px;font-size:11px}
.mecha-tab-native .mecha-badge.ok{background:#dafbe1;color:#1a7f37}
.mecha-tab-native .mecha-badge.warn{background:#fff8c5;color:#9a6700}
.mecha-tab-native .mecha-badge.error{background:#ffebe9;color:#cf222e}
.mecha-tab-native .mecha-badge.ai{background:#fbefff;color:#8250df}
.mecha-tab-native .mecha-page{padding:8px}
.mecha-tab-native .mecha-msg{padding:8px;color:#cf222e}
.mecha-tab-native .mecha-msg.muted{color:#57606a}
.mecha-tab-native .mecha-hint{padding:2px 8px;color:#57606a}
.mecha-tab-native .mecha-headline{padding:4px 8px;font-weight:600}
.mecha-tab-native details.mecha-ev{border:1px solid #d0d7de;border-radius:6px;background:#fff;margin:0 0 4px}
.mecha-tab-native details.mecha-ev>summary{cursor:pointer;padding:4px 8px;color:#1f2328;list-style:none;display:flex;gap:8px}
.mecha-tab-native details.mecha-ev>summary::-webkit-details-marker{display:none}
.mecha-tab-native details.mecha-ev>summary::before{content:'▸';color:#8b949e}
.mecha-tab-native details.mecha-ev[open]>summary::before{content:'▾'}
.mecha-tab-native details.mecha-ev .seq{color:#0969da}
.mecha-tab-native details.mecha-ev .actor-ai{color:#8250df}
.mecha-tab-native details.mecha-ev .mecha-tag{color:#57606a;border:1px solid #d0d7de;border-radius:4px;padding:0 4px;font-size:11px}
.mecha-tab-native details.mecha-ev .mecha-reason{color:#1f2328;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mecha-tab-native details.mecha-ev .body i{color:#8b949e;font-style:normal;margin-right:4px}
.mecha-tab-native details.mecha-ev .body{padding:4px 8px 8px 22px;border-top:1px solid #eaeef2;color:#57606a;white-space:pre-wrap}
.mecha-tab-native details.mecha-sub{margin:2px 0 6px 8px;border-left:2px solid #d0d7de;padding-left:8px}
.mecha-tab-native details.mecha-fam{margin:0 0 6px}
.mecha-tab-native summary{cursor:pointer;color:#57606a;font-weight:600;list-style:none}
.mecha-tab-native summary::-webkit-details-marker{display:none}
.mecha-tab-native summary::before{content:'▸ ';color:#0969da}
.mecha-tab-native details[open]>summary::before{content:'▾ '}
.mecha-tab-native table{width:100%;border-collapse:collapse}
.mecha-tab-native td{border-bottom:1px solid #eaeef2;padding:2px 4px;vertical-align:top}
.mecha-tab-native td:first-child{color:#0550ae}
.mecha-tab-native td.muted{color:#8b949e}
.mecha-tab-native h4{margin:0 0 6px;color:#0969da;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
`
