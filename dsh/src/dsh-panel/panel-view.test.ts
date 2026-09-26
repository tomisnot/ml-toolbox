/**
 * 视图层单测（**共享测试**：随资产一起被复制）。
 *
 * 跑：`node --test mecha/dsh-panel/panel-view.test.ts`
 *
 * 这一层是**纯字符串生成** ⇒ 可以把"**面板不空白**"钉成**字符串级**的性质：
 * 每个分支都必须返回**非空且含可读文本**的 HTML，而不是空串。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  badgeHtml,
  cfgPageHtml,
  esc,
  extraPageHtml,
  fmtVal,
  recPageHtml,
  tabsHtml,
  TAB_CSS,
  type Ev,
} from './panel-view.ts'

const ev = (seq: number, over: Partial<Ev> = {}): Ev => ({
  seq, kind: 'set', actor: 'ai', target: 'field.x', before: 1, after: 2,
  reason: '扫描', call_id: 'call-1', ...over,
})

test('esc：五个危险字符都转义（注入点必须过它）', () => {
  assert.equal(esc('<a href="x">&'), '&lt;a href=&quot;x&quot;&gt;&amp;')
  assert.equal(esc(null), '')
})

test('fmtVal：空值 / 指针 / 截断 / 长 JSON / 标量', () => {
  assert.equal(fmtVal(null), '∅')
  assert.equal(fmtVal(undefined), '∅')
  assert.equal(fmtVal({ $pointer: true, shape: [3, 4] }), '⟨数组 3×4⟩')
  assert.equal(fmtVal({ $truncated: true, kept: 512, total: 4096 }), '⟨截断 512/4096⟩')
  assert.equal(fmtVal('ok'), 'ok')
  assert.equal(fmtVal(7), '7')
  const long = fmtVal({ a: 'x'.repeat(200) })
  assert.ok(long.length <= 120, `长 JSON 应被截断，实得 ${long.length}`)
  assert.ok(long.endsWith('…'))
})

test('⭐ recPageHtml：展开态**换了结果就变**（R17：注入的缝必须真的生效）', () => {
  const evs = [ev(1)]
  const closed = recPageHtml(evs, () => false)
  const opened = recPageHtml(evs, () => true)
  assert.notEqual(closed, opened, '换掉 isOpen 结果没变 ⇒ 这个缝是死的')
  assert.ok(!closed.includes('" open>'), '闭合时不该有 open')
  assert.ok(opened.includes('" open>'), '展开时应当有 open')
})

test('recPageHtml：每条含五要素（seq/actor/kind/reason/target+before→after+call_id）', () => {
  const html = recPageHtml([ev(42, { actor: 'human', reason: '', after: { $truncated: true, kept: 1, total: 2 } })],
    () => false)
  for (const marker of ['#42', 'human', 'set', 'field.x', 'before → after',
                        '⟨截断 1/2⟩', 'call_id', 'call-1']) {
    assert.ok(html.includes(marker), `缺 ${marker}：${html.slice(0, 200)}`)
  }
})

test('recPageHtml：空表 ⇒ **可读文案**（不是空串、也不是空白）', () => {
  const html = recPageHtml([], () => false)
  assert.ok(html.length > 0)
  assert.match(html, /暂无写事件/)
})

test('cfgPageHtml：族→前缀二级树 + 计数 + (top) + orphans 表', () => {
  const html = cfgPageHtml({
    has_schema: true,
    groups: { field: [{ prefix: 'field', rows: [{ name: 'field.x', short: 'x', value: 1 }] },
                      { prefix: '', rows: [] }] },
    orphans: [{ name: 'zzz', short: 'zzz', value: 2, set: { actor: 'ai', seq: 9 } }],
  }, () => false)
  for (const marker of ['field', 'x', '(top)', '契约外（orphans）', 'zzz', 'ai #9']) {
    assert.ok(html.includes(marker), `缺 ${marker}：${html.slice(0, 240)}`)
  }
})

test('cfgPageHtml：没 schema ⇒ **可读说明**，不是一棵空树；且展开态同样随谓词变', () => {
  const noSchema = cfgPageHtml({ has_schema: false, groups: {} }, () => false)
  assert.match(noSchema, /无 schema/)
  const c = cfgPageHtml({ has_schema: true, groups: { f: [{ prefix: 'f', rows: [] }] } }, () => false)
  const o = cfgPageHtml({ has_schema: true, groups: { f: [{ prefix: 'f', rows: [] }] } }, () => true)
  assert.notEqual(c, o)
  assert.ok(o.includes('" open>'))
})

test('tabsHtml / badgeHtml：当前页有 on 标记；徽标类名原样带出', () => {
  const html = tabsHtml([{ id: 'rec', label: '甲' }, { id: 'cfg', label: '乙' }], 'cfg',
    badgeHtml('error', '权威离线'))
  assert.match(html, /data-page="rec"/)
  assert.match(html, /data-page="cfg" class="on">乙/)
  assert.match(html, /class="mecha-badge error">权威离线</)
})

test('TAB_CSS：只用**中性前缀**（共享资产里不许有项目前缀）', () => {
  assert.ok(TAB_CSS.includes('.mecha-tab-native'))
  assert.ok(TAB_CSS.includes('.mecha-ev'))
  for (const prefix of ['.mltb', '.re0']) {   // 这两个前缀只作为**反面语料**出现
    assert.ok(!TAB_CSS.includes(prefix), `CSS 里混进了项目前缀 ${prefix}`)
  }
})

// ---------------------------------------------------------------- 项目附加页

/** 一份**中性的**附加页声明（判据自己的词，不是任何项目的词）。 */
const SPEC = {
  id: 'extra',
  label: '附加页',
  path: '/extra?limit=2',
  title: '附加数据',
  columns: [
    { label: '编号', cell: (r: unknown) => String((r as { id?: unknown }).id ?? '') },
    { label: '得分', cell: (r: unknown) => String((r as { score?: unknown }).score ?? '') },
  ],
  emptyText: '这里还没有记录',
  unreadableText: '读不到附加数据',
  summaryLine: (p: unknown) => `共 ${((p as { rows?: unknown[] })?.rows ?? []).length} 条`,
  note: '空不等于没跑过',
}
const okRes = (d: unknown) => ({ ok: true as const, data: d })

test('⭐ extraPageHtml：有行 ⇒ 表头 + 各行 cell；`summaryLine` 生效', () => {
  const html = extraPageHtml(SPEC, okRes({ rows: [{ id: 'r1', score: 9 }, { id: 'r2', score: 8 }] }))
  for (const marker of ['附加数据', '编号', '得分', 'r1', '9', 'r2', '8', '共 2 条', '空不等于没跑过']) {
    assert.ok(html.includes(marker), `缺 ${marker}：${html.slice(0, 240)}`)
  }
})

test('⭐ extraPageHtml：**读不到 ≠ 空表**（这一档必须画可读错误，不是空表）', () => {
  const html = extraPageHtml(SPEC, { ok: false, failure: { tier: 'route', detail: '500' } })
  assert.match(html, /读不到附加数据/)
  assert.match(html, /500/)
  assert.ok(!html.includes('<tbody'), '失败时不该画表')
  const empty = extraPageHtml(SPEC, okRes({ rows: [] }))
  assert.match(empty, /这里还没有记录/)
  assert.notEqual(html, empty, '两种情形必须可区分')
})

test('⭐ extraPageHtml：声明的函数**换了结果就变**（R17：cell / summaryLine 必须是活的）', () => {
  const rows = [{ id: 'r1', score: 9 }]
  const a = extraPageHtml(SPEC, okRes({ rows }))
  const b = extraPageHtml({ ...SPEC, columns: SPEC.columns.slice(0, 1) }, okRes({ rows }))
  assert.notEqual(a, b, '换掉 columns 结果没变 ⇒ 声明是死的')
  const c = extraPageHtml({ ...SPEC, summaryLine: () => '另一句概括' }, okRes({ rows }))
  assert.notEqual(a, c, '换掉 summaryLine 结果没变 ⇒ 声明是死的')
  assert.match(c, /另一句概括/)
})

test('extraPageHtml：每条分支都返回**非空**可读 HTML（不空白）', () => {
  for (const res of [
    okRes({ rows: [{ id: 'x', score: 1 }] }),
    okRes({ rows: [] }),
    okRes({}),                                   // 没有 rows 键 ⇒ 按空处理（不是崩、不是空白）
    { ok: false as const, failure: { tier: 'route' as const, detail: 'x' } },
  ]) {
    const html = extraPageHtml(SPEC, res)
    assert.ok(html.length > 0)
    assert.match(html, /mecha-page/)
  }
})
