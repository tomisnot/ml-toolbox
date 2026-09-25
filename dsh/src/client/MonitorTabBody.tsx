/**
 * ML 监控 —— 右栏 tab body：**EL 驾驶舱呈现 + ML 的第三页**。
 *
 * 移植自 EL `dsh/src/client/MonitorTabBody.tsx`（只读监控面板）：
 *  - **逐字保留**：`TAB_CSS` 那套呈现、`recPage()` 飞行记录仪（折叠=概括、
 *    展开=`before → after` 全量）、`cfgPage()` 配置态二级树、`openSet`/`savedPage`
 *    的跨挂载展开态。
 *  - **改名**：`re0-*` → `mltb-*`（含 CSS 类名）、标题/按钮换 ML 文案。
 *  - **参数化**：原来的 `MONITOR_URL` 是硬编码常量 `http://127.0.0.1:8767`；
 *    现在地址由注入的 `resolveMonitorBase()` 调 host 半 `/ml-monitor-url`
 *    取（来自权威写的运行期描述符，每次启动端口都可能不同）。
 *  - **新增第三页「运行记录」**（ML 独有）：EL 只有两页；ML 的"跑过什么"有两个
 *    来源——History 的 `ml.run` 事件（`/summary.recent_runs`，AI 跑过就一定有）
 *    与磁盘存档 `runs/`（`/runs`，需 `persist=True` 才落盘）。两者并列显示，
 *    并明确写出"存档为空 ≠ 没跑过"，免得人误读。**没有改动前两页的渲染逻辑**。
 *
 * 数据源全部是权威进程的**只读**端点（`POST` 一律 404）。
 */
import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react'

/** 轮询周期（展示刷新，非正确性依赖）。 */
const POLL_MS = 3000

const TAB_CSS = `
.mltb-native{height:100%;overflow:auto;background:#f6f8fa;color:#1f2328;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.mltb-native .mltb-tabs{position:sticky;top:0;display:flex;gap:4px;align-items:center;padding:6px 8px;background:#f6f8fa;border-bottom:1px solid #d0d7de;z-index:2}
.mltb-native .mltb-tabs button{cursor:pointer;border:1px solid #d0d7de;background:#fff;border-radius:6px;padding:3px 10px;font:inherit;color:#57606a}
.mltb-native .mltb-tabs button.on{background:#0969da;border-color:#0969da;color:#fff}
.mltb-native .mltb-badge{margin-left:auto;padding:2px 8px;border-radius:10px;font-size:11px}
.mltb-native .mltb-badge.ok{background:#dafbe1;color:#1a7f37}
.mltb-native .mltb-badge.warn{background:#fff8c5;color:#9a6700}
.mltb-native .mltb-badge.error{background:#ffebe9;color:#cf222e}
.mltb-native .mltb-badge.ai{background:#fbefff;color:#8250df}
.mltb-native .mltb-page{padding:8px}
.mltb-native details.mltb-ev{border:1px solid #d0d7de;border-radius:6px;background:#fff;margin:0 0 4px}
.mltb-native details.mltb-ev>summary{cursor:pointer;padding:4px 8px;color:#1f2328;list-style:none;display:flex;gap:8px}
.mltb-native details.mltb-ev>summary::-webkit-details-marker{display:none}
.mltb-native details.mltb-ev>summary::before{content:'▸';color:#8b949e}
.mltb-native details.mltb-ev[open]>summary::before{content:'▾'}
.mltb-native details.mltb-ev .seq{color:#0969da}
.mltb-native details.mltb-ev .actor-ai{color:#8250df}
.mltb-native details.mltb-ev .mltb-tag{color:#57606a;border:1px solid #d0d7de;border-radius:4px;padding:0 4px;font-size:11px}
.mltb-native details.mltb-ev .mltb-reason{color:#1f2328;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mltb-native details.mltb-ev .body i{color:#8b949e;font-style:normal;margin-right:4px}
.mltb-native details.mltb-ev .body{padding:4px 8px 8px 22px;border-top:1px solid #eaeef2;color:#57606a;white-space:pre-wrap}
.mltb-native details.mltb-sub{margin:2px 0 6px 8px;border-left:2px solid #d0d7de;padding-left:8px}
.mltb-native details.mltb-fam{margin:0 0 6px}
.mltb-native summary{cursor:pointer;color:#57606a;font-weight:600;list-style:none}
.mltb-native summary::-webkit-details-marker{display:none}
.mltb-native summary::before{content:'▸ ';color:#0969da}
.mltb-native details[open]>summary::before{content:'▾ '}
.mltb-native table{width:100%;border-collapse:collapse}
.mltb-native td{border-bottom:1px solid #eaeef2;padding:2px 4px;vertical-align:top}
.mltb-native td:first-child{color:#0550ae}
.mltb-native h4{margin:0 0 6px;color:#0969da;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.mltb-native .mltb-native-msg{padding:8px;color:#cf222e}
.mltb-native .mltb-hint{padding:8px;color:#57606a}
.mltb-native .mltb-headline{padding:6px 8px;color:#1f2328;font-weight:600}
`

type Page = 'rec' | 'cfg' | 'runs'

interface Ev {
  seq: number; kind: string; actor: string; target: string
  before: unknown; after: unknown; reason?: string
}
interface Row { name: string; short: string; value: unknown; unit?: string; set?: { actor: string; seq: number } | null }
interface Sub { prefix: string; rows: Row[] }
interface RunRow {
  run_id?: string; method?: string; dataset_id?: string; ok?: boolean
  primary_metric?: string; score?: unknown; actor?: string; call_id?: string
}
interface ArchiveRow {
  run_id?: string; method?: string; family?: string; dataset?: string
  primary_metric?: string; value?: unknown; elapsed?: unknown; saved_at?: string
  error?: unknown
}

// 展开态与页签跨挂载保留（模块级，与 EL 的实现同效）
const openSet = new Set<string>()
let savedPage: Page = 'rec'

const esc = (x: unknown): string =>
  String(x ?? '').replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string))

// EL 的 fmtVal：截断/指针友好显示，不 dump 转义 JSON（防"乱码"）
const fmtVal = (v: unknown): string => {
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

/** 飞行记录仪：EL `recPage` 逐行同款（折叠=概括、展开=before→after 全量）。 */
function recPage(evs: Ev[]): string {
  return evs.slice(-60).reverse().map((e) => {
    const k = 'ev:' + e.seq
    const open = openSet.has(k) ? ' open' : ''
    return `<details class="mltb-ev" data-key="${k}"${open}><summary>` +
      `<span class="seq">#${e.seq}</span>` +
      `<span class="${e.actor === 'ai' ? 'actor-ai' : ''}">${esc(e.actor)}</span>` +
      `<span class="mltb-tag">${esc(e.kind)}</span>` +
      `<span class="mltb-reason" title="${esc(e.target)}">${esc(e.reason || e.target)}</span>` +
      `</summary><div class="body">` +
      `<div><i>target</i>${esc(e.target)}</div>` +
      `<div><i>before → after</i>${esc(fmtVal(e.before))} → ${esc(fmtVal(e.after))}</div>` +
      `<div><i>reason</i>${esc(e.reason || '')}</div>` +
      `<div><i>call_id</i>${esc((e as any).call_id || '')}</div>` +
      `</div></details>`
  }).join('')
}

/** 配置态：EL `cfgPage` 同款（族→前缀二级 details 树，显示名去前缀）。 */
function cfgPage(groups: Record<string, Sub[]>, orphans: Row[]): string {
  const tree = Object.entries(groups || {}).map(([fam, subs]) =>
    `<details class="mltb-fam" data-key="fam:${fam}"${openSet.has('fam:' + fam) ? ' open' : ''}>` +
    `<summary>${esc(fam)}</summary>` +
    (subs || []).map((sg) =>
      `<details class="mltb-sub" data-key="sub:${fam}/${sg.prefix}"` +
      `${openSet.has('sub:' + fam + '/' + sg.prefix) ? ' open' : ''}>` +
      `<summary>${esc(sg.prefix || '(top)')} (${(sg.rows || []).length})</summary><table>` +
      (sg.rows || []).map((r) =>
        `<tr><td title="${esc(r.name)}${r.set ? ' ← ' + esc(r.set.actor) + ' #' + r.set.seq : ''}">` +
        `${esc(r.short || r.name)}</td>` +
        `<td>${esc(fmtVal(r.value))}${r.unit ? ' <i>' + esc(r.unit) + '</i>' : ''}</td></tr>`).join('') +
      '</table></details>').join('') +
    '</details>').join('')
  // ML 走"无 schema"分支：EL 的提示文案 + orphans 表（如实投影，不造树）
  const noSchema = `<div class="mltb-native-msg" style="color:#57606a">` +
    `无 schema（ML 没有逐键 schema 面）——下方按当前快照直接列出现值</div>`
  const orphanTable = (orphans || []).length
    ? `<h4>契约外（orphans）</h4><table>${orphans.map((r) =>
        `<tr><td title="${esc(r.name)}">${esc(r.name)}</td><td>${esc(fmtVal(r.value))}` +
        `</td><td style="color:#8b949e">${r.set ? esc(r.set.actor) + ' #' + r.set.seq : ''}</td></tr>`
      ).join('')}</table>`
    : ''
  return tree + noSchema + orphanTable
}

/** 运行记录（ML 独有第三页）：History 最近的运行 + 磁盘存档，两个来源并列。 */
function runsPage(summary: Record<string, any>, archived: ArchiveRow[]): string {
  const recent: RunRow[] = (summary?.recent_runs as RunRow[]) || []
  const headline = String(summary?.headline || '')
  const disputed = summary?.disputed
    ? `<div class="mltb-native-msg" style="color:#9a6700">概括与原始史不一致，原始赢：` +
      `${esc(summary?.dispute_reason || '')}</div>`
    : ''
  const head = headline ? `<div class="mltb-headline">${esc(headline)}</div>` : ''
  const counts = `<div class="mltb-hint">运行 ${esc(summary?.run_count ?? 0)} 次 · ` +
    `成功 ${esc(summary?.ok_count ?? 0)} · 失败 ${esc(summary?.failed_count ?? 0)}` +
    `（来自操作史 ml.run 事件：AI 跑过就一定在这里）</div>`
  const recentTable = recent.length
    ? `<h4>最近运行（操作史）</h4><table>` + recent.map((r) =>
        `<tr><td>${esc(r.run_id)}</td><td>${esc(r.method)}</td>` +
        `<td>${esc(r.primary_metric)}${r.score === undefined || r.score === null ? '' : '=' + esc(r.score)}</td>` +
        `<td style="color:#8b949e">${esc(r.actor)}${r.ok === false ? ' · 失败' : ''}</td></tr>`).join('') +
      `</table>`
    : `<div class="mltb-hint">本次会话还没有运行记录。</div>`
  const archiveTable = archived.length
    ? `<h4>磁盘存档 runs/（跨重启可见）</h4><table>` + archived.map((r) =>
        `<tr><td>${esc(r.run_id)}</td><td>${esc(r.method)}</td>` +
        `<td>${esc(r.primary_metric)}=${esc(r.value)}</td>` +
        `<td style="color:#8b949e">${esc(r.saved_at)}${r.error ? ' · ' + esc(r.error) : ''}</td></tr>`).join('') +
      `</table>`
    : `<div class="mltb-hint">磁盘上没有落盘的运行。**注意：存档为空不等于没跑过**——` +
      `运行只在 persist=True（或 GUI 批量成功）时写入 runs/。</div>`
  return head + disputed + counts + recentTable + archiveTable
}

export function MonitorTabBody({ resolveMonitorBase }: {
  resolveMonitorBase?: () => Promise<string>
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [page, setPage] = useState<Page>(savedPage)
  const [base, setBase] = useState<string>('')
  const [offline, setOffline] = useState(false)
  const [resolving, setResolving] = useState(true)
  const [resolveError, setResolveError] = useState('')
  const [mode, setMode] = useState('')
  const [events, setEvents] = useState<Ev[]>([])
  const [groups, setGroups] = useState<Record<string, Sub[]>>({})
  const [orphans, setOrphans] = useState<Row[]>([])
  const [summary, setSummary] = useState<Record<string, any>>({})
  const [archived, setArchived] = useState<ArchiveRow[]>([])

  // 地址解析：**不硬编码**——问 host 半要（地址来自权威的运行期描述符）。
  const resolve = useCallback(async () => {
    if (!resolveMonitorBase) {
      setResolving(false)
      setResolveError('面板缺少监控地址解析器（插件 host 半没挂上？）')
      return
    }
    setResolving(true)
    try {
      const url = await resolveMonitorBase()
      setBase(url)
      setResolveError('')
    } catch (err) {
      setBase('')
      setResolveError((err as Error)?.message || String(err))
    } finally {
      setResolving(false)
    }
  }, [resolveMonitorBase])

  useEffect(() => { void resolve() }, [resolve])

  // 轮询：/history（写事件，含 mode）+ /config（结构树）+ /summary + /runs。
  useEffect(() => {
    if (!base) return
    let dead = false
    const poll = async () => {
      try {
        const [h, c, s, r] = await Promise.all([
          fetch(`${base}/history?since_seq=0`).then((x) => x.json()),
          fetch(`${base}/config`).then((x) => x.json()),
          fetch(`${base}/summary`).then((x) => x.json()),
          fetch(`${base}/runs?limit=50`).then((x) => x.json()),
        ])
        if (dead) return
        setOffline(false)
        setMode(h.mode ?? '')
        setEvents(h.events ?? [])
        setGroups(c.groups ?? {})
        setOrphans(c.orphans ?? [])
        setSummary({ ...(s.summary ?? {}), disputed: s.disputed,
                     dispute_reason: s.dispute_reason })
        setArchived(r.runs ?? [])
      } catch {
        if (!dead) setOffline(true)
      }
    }
    void poll()
    const t = setInterval(poll, POLL_MS)
    return () => { dead = true; clearInterval(t) }
  }, [base])

  // CSS 注入一次（data 属性防重）
  useEffect(() => {
    if (!document.querySelector('style[data-mltb-tab-css]')) {
      const s = document.createElement('style')
      s.setAttribute('data-mltb-tab-css', '1')
      s.textContent = TAB_CSS
      document.head.appendChild(s)
    }
  }, [])

  // 事件委托：页签切换 + 记录展开态（React 委托挂在容器上，对
  // dangerouslySetInnerHTML 重建的子树同样生效）
  const onClick = (e: MouseEvent) => {
    const tgt = e.target as HTMLElement
    const pg = tgt.closest?.('[data-page]')
    if (pg) {
      const p = pg.getAttribute('data-page') as Page
      savedPage = p
      setPage(p)
      return
    }
    const sm = tgt.closest?.('summary')
    const dt = sm?.parentElement
    const key = dt?.getAttribute?.('data-key')
    if (key) { if (openSet.has(key)) openSet.delete(key); else openSet.add(key) }
  }

  if (resolving || !base) {
    return (
      <div ref={ref} className="mltb-native">
        <div className="mltb-hint">
          {resolving ? '正在解析监控端点地址…' : `监控端点未知：${resolveError}`}
          {!resolving && (
            <div style={{ marginTop: 8 }}>
              <button type="button" onClick={() => void resolve()}>重试</button>
              <div style={{ color: '#8b949e', marginTop: 4 }}>
                监控端点只在 AI 模式存在；地址由软件写进运行期描述符，面板**不猜**端口。
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  const badge = offline
    ? `<span class="mltb-badge error">权威离线</span>`
    : `<span class="mltb-badge ${mode === 'ai' ? 'ai' : mode === 'locked' ? 'warn' : 'ok'}">` +
      `${esc(mode === 'ai' ? '写权 AI' : mode === 'locked' ? '全员禁写' : mode === 'human' ? '写权在人' : '连接中…')}</span>`
  const tabs = `<div class="mltb-tabs">` +
    `<button data-page="rec" class="${page === 'rec' ? 'on' : ''}">飞行记录仪</button>` +
    `<button data-page="cfg" class="${page === 'cfg' ? 'on' : ''}">配置态</button>` +
    `<button data-page="runs" class="${page === 'runs' ? 'on' : ''}">运行记录</button>` +
    badge + `</div>`
  const body = page === 'rec'
    ? `<div class="mltb-page">${events.length ? recPage(events) :
        `<div class="mltb-hint" style="color:#57606a">${offline ? '连不上权威（软件没在 AI 模式跑？）' : '暂无写事件——AI/人每改一个键都会出现在这里'}</div>`}</div>`
    : page === 'cfg'
      ? `<div class="mltb-page"><h4>配置态</h4>${cfgPage(groups, orphans)}</div>`
      : `<div class="mltb-page"><h4>运行记录</h4>${runsPage(summary, archived)}</div>`

  return (
    <div ref={ref} onClick={onClick}
      // 数据变化后由 React 重渲染 innerHTML
      dangerouslySetInnerHTML={{ __html: `<div class="mltb-native">${tabs}${body}</div>` }} />
  )
}
