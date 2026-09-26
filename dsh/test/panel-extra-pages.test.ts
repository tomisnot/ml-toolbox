/**
 * ML 附加页声明的判据（**项目侧**）：数据层可 node 直测（本文件与 `extraPages.ts`
 * 运行时零 react 依赖）。
 *
 * 覆盖四件事：
 *  1. 两条页的声明形状（id / 路径 / 文案分工）；
 *  2. **`emptyText` 与 `unreadableText` 必须不同**（D6：空表 ≠ 读不到）；
 *  3. 取值的**两个来源字段名不同**（操作史行 `score`、磁盘行 `metrics[主指标]`）——
 *     旧自绘面板只读 `r.value`，磁盘行渲染成 `f1=`（值丢了）；
 *  4. 用**资产的** `extraPageHtml` 真渲染一遍：有行 → 表格；**无行但有数据 → 走
 *     emptyText 分支**（实测结论，正是 `/summary` 也要回 `rows` 的原因）；失败 → 可读错误。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EXTRA_PAGES, metricCell, savedAtCell, summaryLine } from '../src/client/extraPages.ts'
import { extraPageHtml } from '../src/dsh-panel/panel-view.ts'

const byId = (id: string) => {
  const spec = EXTRA_PAGES.find((p) => p.id === id)
  assert.ok(spec, `缺少附加页 ${id}`)
  return spec!
}

test('两条附加页：id / 路径 / 页签文案（一页一条路由）', () => {
  assert.deepEqual(EXTRA_PAGES.map((p) => p.id), ['summary', 'runs'])
  assert.deepEqual(EXTRA_PAGES.map((p) => p.label), ['运行记录', '磁盘存档'])
  // `summary` 不需要查询串；`runs` 的 limit 属于**请求路径**（不参与任何索引）
  assert.equal(byId('summary').path, '/summary')
  assert.equal(byId('runs').path, '/runs?limit=50')
  for (const p of EXTRA_PAGES) {
    assert.ok(p.title && p.columns.length >= 3, `${p.id} 的标题/列不完整`)
  }
})

test('⭐ emptyText 与 unreadableText 必须不同（空表 ≠ 读不到）', () => {
  for (const p of EXTRA_PAGES) {
    assert.notEqual(p.emptyText, p.unreadableText, `${p.id} 两档文案相同 ⇒ 无法区分`)
    assert.ok(p.emptyText.trim() && p.unreadableText.trim(), `${p.id} 文案为空`)
  }
  assert.match(byId('runs').unreadableText, /不是空的/, '磁盘存档的失败文案必须与"空表"区分')
})

test('⭐ 主指标列：两个来源的字段名不同（旧面板在磁盘行上丢过值）', () => {
  // 操作史行（`/summary.rows`）：值在 `score`
  assert.equal(metricCell({ primary_metric: 'f1', score: 0.9 }), 'f1=0.9')
  // 磁盘行（`/runs.rows`）：值在 `metrics[主指标]`，**没有** `score`
  assert.equal(metricCell({ primary_metric: 'rmse', metrics: { rmse: 1.25 } }), 'rmse=1.25')
  // 两处都缺 ⇒ 只给指标名，不渲染 `undefined`
  assert.equal(metricCell({ primary_metric: 'f1' }), 'f1')
  assert.equal(metricCell({ primary_metric: 'f1', metrics: {} }), 'f1')
  // 时间列：错误摘要带上（不吞）
  assert.equal(savedAtCell({ saved_at: 't', error: 'boom' }), 't · boom')
})

test('⭐ 概括一行：次数 + 对账结论（`disputed` 由项目承担，共享标准区不渲染它）', () => {
  const ok = summaryLine({ summary: { run_count: 3, ok_count: 3, failed_count: 0 } })
  assert.match(ok, /运行 3 次 · 成功 3 · 失败 0/, ok)
  assert.match(ok, /概括与原始史一致/, ok)
  const bad = summaryLine({ summary: { run_count: 2, ok_count: 1, failed_count: 1 },
                            disputed: true, dispute_reason: '命令数对不上' })
  assert.match(bad, /⚠ 概括与原始史不一致，原始赢：命令数对不上/, bad)
  // 缺字段不崩、不渲染 undefined
  assert.match(summaryLine({}), /运行 0 次/, summaryLine({}))
})

test('⭐ 真渲染：有行 → 表格；**无行但有数据 → 走 emptyText**（实测结论）', () => {
  const runs = byId('runs')
  const okRows = { ok: true as const, data: { rows: [
    { run_id: 'r1', method: 'ridge', primary_metric: 'rmse', metrics: { rmse: 1.5 },
      saved_at: '2026-09-25' }], source: 'runs/' } }
  const html = extraPageHtml(runs, okRows)
  assert.match(html, /磁盘存档 runs\//, html)
  assert.match(html, /r1/, '表格里没有行数据')
  assert.match(html, /rmse=1\.5/, '主指标值没渲染出来（磁盘行丢值的旧病）')
  assert.match(html, /主指标/, '表头缺列名')

  // ⭐ 实测结论（2026-09-26，ab28452 资产）：**无 rows 的载荷**
  //   * `summaryLine` **会**被渲染（`headLine`），所以"概括丢不丢"不是问题；
  //   * **但 `emptyText` 也会被画**（`if (!rows.length)` 那条分支不看 payload 有没有数据）
  //     ⇒ 于是"运行 7 次"和"还没有运行记录"同时出现，**自相矛盾**。
  // ⇒ 这就是本仓让 `/summary` **也回 `rows`**（= 操作史 recent_runs）的原因：既拿到概括行，
  //   又让表格真有行、不落进那条矛盾分支。**这是绕开，不是改资产**（资产零改动）。
  const noRows = { ok: true as const, data: { summary: { run_count: 7 }, disputed: false } }
  const htmlNoRows = extraPageHtml(byId('summary'), noRows)
  assert.match(htmlNoRows, /运行 7 次/, '无 rows 时 summaryLine 仍应渲染（实测）')
  assert.match(htmlNoRows, /还没有运行记录/,
    '无 rows 时那条 emptyText 也会画（实测）⇒ 与上一行自相矛盾，故本仓改为回 rows')

  // 失败档：可读错误 + 真因，**不是空表**
  const bad = extraPageHtml(runs, { ok: false as const,
                                    failure: { tier: 'route' as const, detail: '磁盘读不到' } })
  assert.match(bad, /不是空的/, bad)
  assert.match(bad, /磁盘读不到/, bad)
  assert.ok(!/<table/.test(bad), '失败档不该画空表')
})

test('R17：列 cell 与 summaryLine 是**活的**（换掉它们输出就变）', () => {
  const runs = byId('runs')
  const payload = { ok: true as const, data: { rows: [{ run_id: 'X', method: 'm',
                                                        primary_metric: 'p', metrics: { p: 1 } }] } }
  // 同一份数据，换一个 cell ⇒ 输出必须变（证明 cell 真的被调用，不是装饰）
  const patched = { ...runs, columns: [{ label: 'L', cell: () => 'SENTINEL' }] }
  const html = extraPageHtml(patched as never, payload)
  assert.match(html, /SENTINEL/, '声明的 cell 没被调用 ⇒ 那是死缝')
})
