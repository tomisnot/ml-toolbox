/**
 * ML 附加页的判据（**项目侧**）：**呈现细化**（本仓仍拥有的那一半）。
 *
 * ## 阶段② 起这份判据守什么（承诺换人了，逐条交代）
 *
 * | 原判据守的 | 现在由谁守 |
 * |---|---|
 * | 两条页的 id / 路径 / 页签文案 | ⭐ **声明**（`ml_mecha/mecha_site.py` 的 `ExtraPage`）+ 框架判据把**夹具**与真声明对拍（`checks/phase13_panel_decl_selfcheck.py`） |
 * | `emptyText` 与 `unreadableText` 必须不同 | 同上（声明里的两个字段）+ 资产的中性缺省也把两者分开 |
 * | 列的**取值**（主指标按来源各取一次、时间带错误摘要、概括行） | ⭐ **本文件**（呈现细化 = 代码，见 `extraPages.ts`） |
 * | 用资产的 `extraPageHtml` 真渲染 | **本文件**（拿"声明的数据 + 本仓的呈现"合出来的 spec 渲染） |
 *
 * ⚠ 页集/路径那几条**不再由本文件断言字面量**（那是第二处真值），改为**对拍声明原文**
 * （下一条用例）—— 一句话：字面量只许住声明里。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { extraPageHtml, type ExtraPageSpec } from '@mecha/dsh-panel/panel/panel-view.ts'
import {
  contributeExtraPageRender,
  extraPageRenderKey,
  extraPageSpecsFrom,
} from '@mecha/dsh-panel/panel/services.ts'

import { ML_EXTRA_PAGE_RENDERS, metricCell, savedAtCell, summaryLine } from '../src/client/extraPages.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
/** 项目的站点声明（面板参数的**家**）。 */
const DECL_FILE = join(HERE, '..', '..', 'ml_mecha', 'mecha_site.py')

/**
 * ⭐ **声明的数据 + 本仓的呈现**合出来的 spec —— 正是框架在浏览器里做的那件事。
 *
 * ⚠ 页数据这里按**声明原文**给（并与原文对拍，见下一条用例）：本判据验的是"呈现细化
 * 接上去之后渲染对不对"，不是"声明写了什么"。
 */
const DECLARED_PAGES = [
  {
    id: 'summary', label: '运行记录', path: '/summary', title: '运行记录（操作史 ml.run 事件）',
    emptyText: '本次会话还没有运行记录——AI/人每跑一次方法都会出现在这里。',
    unreadableText: '运行记录读取失败：这一格不是「没有运行」，是没读到',
    note: '来自操作史 ml.run 事件：AI 跑过就一定在这里（与磁盘存档是两件事）。',
    columns: [{ label: '运行编号', field: 'run_id' }, { label: '方法', field: 'method' },
              { label: '主指标', field: 'primary_metric' }, { label: '操作者', field: 'actor' }],
  },
  {
    id: 'runs', label: '磁盘存档', path: '/runs?limit=50', title: '磁盘存档 runs/（跨重启可见）',
    emptyText: '磁盘上没有落盘的运行。**注意：存档为空不等于没跑过**——'
      + '运行只在 persist=True（或 GUI 批量成功）时写入 runs/。',
    unreadableText: '磁盘存档读取失败：这一格**不是空的**，是没读到（权威的存档读取出错了）',
    note: '跨重启可见；与「运行记录」是两件事（那份来自操作史，这份来自磁盘）。',
    columns: [{ label: '运行编号', field: 'run_id' }, { label: '方法', field: 'method' },
              { label: '主指标', field: 'primary_metric' }, { label: '时间', field: 'saved_at' }],
  },
]

/** 注册本仓那两份呈现细化（**与 `client/index.ts` 同一件事**），返回 spec。
 *  ⚠ 为什么不能直接 import `client/index.ts`：它静态 import 资产**入口** ⇒ 会把 `.tsx` 壳
 *  拖进来，而 node 加载不了 `.tsx`（资产既有边界）⇒ 这里照同一件事做一遍。 */
function specs(): ExtraPageSpec[] {
  const ids = Object.keys(ML_EXTRA_PAGE_RENDERS)
  assert.deepEqual(ids.sort(), ['runs', 'summary'], `呈现细化应当只覆盖这两页：${ids}`)
  for (const [pageId, render] of Object.entries(ML_EXTRA_PAGE_RENDERS)) {
    contributeExtraPageRender(extraPageRenderKey('ML', pageId), render)
  }
  return extraPageSpecsFrom('ML', DECLARED_PAGES as never)
}

const byId = (id: string): ExtraPageSpec => {
  const spec = specs().find((p) => p.id === id)
  assert.ok(spec, `缺少附加页 ${id}`)
  return spec!
}

test('⭐ 字面量只许住声明：页集/路径/两条文案都在 mecha_site.py 里', () => {
  const decl = readFileSync(DECL_FILE, 'utf8')
  for (const [what, literal] of [
    ['summary 页', 'id="summary"'],
    ['runs 页', 'id="runs"'],
    ['summary 路径', 'path="/summary"'],
    ['runs 路径', 'path="/runs?limit=50"'],
    ['摘要的读不到文案', 'unreadable_text="运行记录读取失败：这一格不是「没有运行」，是没读到"'],
    ['存档的读不到文案', '这一格**不是空的**，是没读到'],
  ] as const) {
    assert.ok(decl.includes(literal), `声明里没有 ${what}（${literal}）`)
  }
  // 对偶（不许空洞）：这两页**必须在**声明里带列（没有列的表看着像坏了）
  assert.equal((decl.match(/ExtraColumn\(/g) ?? []).length, 8,
    '声明里应当有 8 条 ExtraColumn（两页 × 4 列）')
})

test('⭐ 呈现细化：列来自 `ML_EXTRA_PAGE_RENDERS`，不是声明里的中性取值', () => {
  const summary = byId('summary')
  // ⚠ 中性取值会给空串（声明里的 field 取不到值）；细化后必须给出**项目自己的**渲染
  assert.equal(summary.columns[2]?.cell({ primary_metric: 'f1', score: 0.9 }), 'f1=0.9')
  assert.equal(byId('runs').columns[3]?.cell({ saved_at: 't', error: 'boom' }), 't · boom')
  assert.equal(typeof summary.summaryLine, 'function', 'summary 页的概括行应当由本仓给')
  assert.equal(byId('runs').summaryLine, undefined, 'runs 页本来就没有概括行（别编一个空的）')
})

test('⭐ emptyText 与 unreadableText 必须不同（空表 ≠ 读不到）', () => {
  for (const p of specs()) {
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

test('R17：呈现细化是**活的**（换掉列 ⇒ 输出就变）', () => {
  const runs = byId('runs')
  const payload = { ok: true as const, data: { rows: [{ run_id: 'X', method: 'm',
                                                        primary_metric: 'p', metrics: { p: 1 } }] } }
  // 同一份数据，换一个 cell ⇒ 输出必须变（证明 cell 真的被调用，不是装饰）
  const patched = { ...runs, columns: [{ label: 'L', cell: () => 'SENTINEL' }] }
  const html = extraPageHtml(patched as never, payload)
  assert.match(html, /SENTINEL/, '呈现细化的 cell 没被调用 ⇒ 那是死缝')
})
