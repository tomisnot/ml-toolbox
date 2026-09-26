/**
 * ML 的**附加页声明**（共享面板的 `ExtraPageSpec[]`）：数据层，**无 react、无 DOM**。
 *
 * ## 为什么单独一个文件（与资产 `panel-data.ts` 同一条理由）
 *
 * 1. **判据要落在这一层**：Node 能直接 import 它喂真 payload（本文件只 import `type`，
 *    运行时零依赖 ⇒ `node --test` 直接测，不需要 react —— react 是 dsh 宿主提供的 peer）；
 * 2. **领域词全部收在这里**：共享资产里零 ML 字面量（词表守卫在扫 `.ts`/`.tsx`），
 *    ML 的列名、文案、取值都在本文件 ⇒ "能力并入、领域词不进框架"。
 *
 * ## 数据契约（共享面板定 wire 形状）
 *
 * 附加路由**必须**回 `{ rows: [...] }`：表格行取 `payload.rows`，其余键交给 `summaryLine`。
 * ⚠ **实测边界**：`extraPageHtml` 在 `rows` 为空时**走 emptyText 分支**（把"没有行"当成
 * 空表）——所以**没有行但有数据的**载荷（如 `/summary` 的概括信封）不能只给信封：
 * 本仓让 `/summary` 也回 `rows`（= 操作史里的 `recent_runs`），于是"概括一行 + 表格"都在，
 * 既不丢信息也不会被误显示成"没有记录"。
 */
import type { ExtraPageSpec } from '../dsh-panel/panel-view.ts'

/** 取文本（`cell` 用；缺字段给空串，**不渲染 `undefined`**）。 */
export const asText = (v: unknown): string =>
  (v === null || v === undefined ? '' : String(v))

/**
 * 一行里的"主指标"列。
 *
 * ⚠ 两个来源的字段名不同（**实测踩过**）：
 * * `/summary` 的行来自操作史 `ml.run` 事件 ⇒ 值在 `score`；
 * * `/runs` 的行来自磁盘 `record.json` ⇒ 值在 `metrics[primary_metric]`（没有 `score`）。
 * 旧的自绘面板只读 `r.value`，于是磁盘行渲染成 `f1=`（值丢了）——这里按来源各取一次。
 */
export function metricCell(row: unknown): string {
  const r = (row ?? {}) as Record<string, unknown>
  const name = asText(r['primary_metric'])
  let value: unknown = r['score']
  if (value === undefined) {
    const metrics = r['metrics']
    value = (metrics && typeof metrics === 'object')
      ? (metrics as Record<string, unknown>)[name]
      : ''
  }
  return name + (value === '' || value === undefined ? '' : '=' + String(value))
}

/** 操作者列（`/summary` 的行）：带"失败"标注。 */
export function actorCell(row: unknown): string {
  const r = (row ?? {}) as Record<string, unknown>
  return asText(r['actor']) + (r['ok'] === false ? ' · 失败' : '')
}

/** 时间列（`/runs` 的行）：带错误摘要。 */
export function savedAtCell(row: unknown): string {
  const r = (row ?? {}) as Record<string, unknown>
  return asText(r['saved_at']) + (r['error'] ? ' · ' + asText(r['error']) : '')
}

/**
 * `/summary` 的**概括一行**：次数 + 对账结论。
 *
 * ⚠ `disputed` 是权威给的可对账信号（概括层与原始史不一致时**原始赢**）。共享面板的
 * 标准区**不渲染** `/summary` 的信封 ⇒ 这句由本函数承担（项目侧义务，不是资产的洞）。
 */
export function summaryLine(payload: unknown): string {
  const p = (payload ?? {}) as Record<string, any>
  const s = (p['summary'] ?? {}) as Record<string, unknown>
  const line = `运行 ${asText(s['run_count'] ?? 0)} 次 · 成功 ${asText(s['ok_count'] ?? 0)}` +
    ` · 失败 ${asText(s['failed_count'] ?? 0)}`
  return p['disputed']
    ? `${line}　⚠ 概括与原始史不一致，原始赢：${asText(p['dispute_reason'])}`
    : `${line}　（概括与原始史一致）`
}

/**
 * ML 的两条附加页（**声明式**）。
 *
 * ⚠ **一页一条路由**（资产机制）⇒ 旧面板里"操作史与磁盘存档并列一页"变成**两页**
 * （已申报的行为变更；功能不丢）。
 */
export const EXTRA_PAGES: ExtraPageSpec[] = [
  {
    id: 'summary',
    label: '运行记录',
    path: '/summary',
    title: '运行记录（操作史 ml.run 事件）',
    columns: [
      { label: '运行编号', cell: (r) => asText((r as any)?.run_id) },
      { label: '方法', cell: (r) => asText((r as any)?.method) },
      { label: '主指标', cell: metricCell },
      { label: '操作者', cell: actorCell },
    ],
    summaryLine,
    emptyText: '本次会话还没有运行记录——AI/人每跑一次方法都会出现在这里。',
    unreadableText: '运行记录读取失败：这一格不是「没有运行」，是没读到',
    note: '来自操作史 ml.run 事件：AI 跑过就一定在这里（与磁盘存档是两件事）。',
  },
  {
    id: 'runs',
    label: '磁盘存档',
    path: '/runs?limit=50',
    title: '磁盘存档 runs/（跨重启可见）',
    columns: [
      { label: '运行编号', cell: (r) => asText((r as any)?.run_id) },
      { label: '方法', cell: (r) => asText((r as any)?.method) },
      { label: '主指标', cell: metricCell },
      { label: '时间', cell: savedAtCell },
    ],
    emptyText:
      '磁盘上没有落盘的运行。**注意：存档为空不等于没跑过**——' +
      '运行只在 persist=True（或 GUI 批量成功）时写入 runs/。',
    // ⚠「读不到 ≠ 空表」：真因摆成人能读的文本（旧面板在这一点上的措辞原样保留）。
    unreadableText:
      '磁盘存档读取失败：这一格**不是空的**，是没读到（权威的存档读取出错了）',
    note: '跨重启可见；与「运行记录」是两件事（那份来自操作史，这份来自磁盘）。',
  },
]
