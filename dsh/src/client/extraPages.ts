/**
 * ML 的**附加页呈现细化**（"自定义 UI"那一半；数据层，**无 react、无 DOM**）。
 *
 * ## 阶段② 起本文件的**职责缩小了**（这是刻意的）
 *
 * | 项 | 现在谁拥有 |
 * |---|---|
 * | 页集 / 页签文案 / 请求路径 / 页内标题 / 空表文案 / 读不到文案 / 脚注 | ⭐ **声明**（`ml_mecha/mecha_site.py` 的 `ExtraPage`） |
 * | 列名与"取哪个字段" | ⭐ **声明**（`ExtraColumn(label, field)`） |
 * | 列的**取值函数**（`metricCell` 等）与概括行 | **本文件**（代码 —— JSON 装不下，故走"自定义 UI"那条路） |
 *
 * ⇒ 于是"一页一条路由""空表 ≠ 读不到"那些**声明性**的事由声明与它的判据管，
 * 本文件只管**怎么把一行渲染得好看**。⚠ 两处**不许重叠**（重叠就是两份真值）。
 *
 * ## 为什么还留着（不是没删干净）
 *
 * 主指标列有**真 bug 的历史**：两个来源的字段名不同（操作史行在 `score`、磁盘行在
 * `metrics[主指标]`），旧自绘面板只读 `r.value` ⇒ **磁盘行渲染成 `f1=`（值丢了）**。
 * 那份按来源各取一次的逻辑是**代码**，进不了声明 ⇒ 只能住这里。
 *
 * ## 数据契约（共享面板定 wire 形状）
 *
 * 附加路由**必须**回 `{ rows: [...] }`：表格行取 `payload.rows`，其余键交给 `summaryLine`。
 * ⚠ **实测边界**：`extraPageHtml` 在 `rows` 为空时**走 emptyText 分支**（把"没有行"当成空表）
 * ——所以**没有行但有数据的**载荷（如 `/summary` 的概括信封）不能只给信封：
 * 本仓让 `/summary` 也回 `rows`（= 操作史里的 `recent_runs`），于是"概括一行 + 表格"都在。
 */
import type { ExtraPageRender } from '@mecha/dsh-panel/panel/auto.ts'

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
 * ⭐ 本项目的**呈现细化表**（键 = 页 id；⚠ 注册时还要带上服务 id，见 `client/index.ts`）。
 *
 * ⚠ **只放"怎么渲染"**：页集/路由/文案/列名都在声明里。
 * ⚠ `runs` 那份**不给 `summaryLine`**：它的概括在声明里本来就没有（旧面板也没有）——
 * 不给就是不给，别编一个空的（`extraPageSpecsFrom` 只在**有**的时候才挂上去）。
 */
export const ML_EXTRA_PAGE_RENDERS: Record<string, ExtraPageRender> = {
  summary: {
    columns: [
      { label: '运行编号', cell: (r) => asText((r as any)?.run_id) },
      { label: '方法', cell: (r) => asText((r as any)?.method) },
      { label: '主指标', cell: metricCell },
      { label: '操作者', cell: actorCell },
    ],
    summaryLine,
  },
  runs: {
    columns: [
      { label: '运行编号', cell: (r) => asText((r as any)?.run_id) },
      { label: '方法', cell: (r) => asText((r as any)?.method) },
      { label: '主指标', cell: metricCell },
      { label: '时间', cell: savedAtCell },
    ],
  },
}
