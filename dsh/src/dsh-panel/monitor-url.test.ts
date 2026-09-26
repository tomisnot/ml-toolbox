/**
 * 监控**地址**纯逻辑单测（**共享测试**：随参考实现一起被复制）。
 *
 * 跑：`node --test mecha/dsh-panel/monitor-url.test.ts`
 * （Node ≥ 22.6 原生剥类型，**不需要任何 npm 依赖**；老版本加 `--experimental-strip-types`）
 *
 * 钉死的三条语义：
 *  1. 端口文件读得到 ⇒ 返回 `http://127.0.0.1:<port>`；
 *  2. ⭐ **读不到 / 内容非法 / 越界 / 文件名没配 ⇒ 抛错，绝不回落任何默认端口或默认文件名**
 *     —— 回落会把"权威没起来"显示成"连上了但空面板"（本工程反复在防的假绿）；
 *  3. **每次调用现读**（端口漂移自愈）。
 * 外加路由处理器的状态码语义：成功 `200 {base}`、失败 `503` + 可读错误（且**没有 `base` 字段**）。
 *
 * ⚠ **本文件不许出现任何项目字面量**（项目名、历史端口、端口文件名…）——它是共享件。
 * 项目要加"自家历史默认值"这类**反面语料**，请另建一个**项目自己的**测试文件，
 * **不要改这里**（改了就不是逐字一致了）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  makeMonitorUrlHandler,
  resolveMonitorBase,
  sendJson,
} from './monitor-url.ts'
// ⚠ 浏览器安全的常量住 `routes.ts`（client 半也要用它）；`monitor-url.ts` 是 **node-only**。
import { BASIC_ROUTES, DEFAULT_ROUTE_PATH } from './routes.ts'

const PORT_FILE = 'probe.port'   // 中性名：判据自己造的文件，不是任何项目的约定名

/** 最小 res 替身：只记状态码/头/正文。
 *
 * ⚠ 正文**如实解码成字符串**（真客户端看到的就是字符串）——若这里原样存 `Buffer`，
 * `res.body.length` 会等于**字节数**，于是"Content-Length 按字节算"那条断言会
 * 变成同一律（恒真、不携带信息）。
 */
function fakeRes() {
  const res: any = {
    statusCode: 0,
    headers: {} as Record<string, string>,
    body: '',
    setHeader(k: string, v: string) { res.headers[k] = v },
    end(b: string | Uint8Array) {
      res.body = typeof b === 'string' ? b : Buffer.from(b).toString('utf8')
    },
  }
  return res
}

function withTmp(fn: (dir: string) => void): void {
  const dir = mkdtempSync(join(tmpdir(), 'mecha-panel-'))
  try {
    fn(dir)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

test('默认路由是**中性**的（不含任何项目名）', () => {
  assert.equal(DEFAULT_ROUTE_PATH, '/mecha/monitor-url')
  assert.ok(DEFAULT_ROUTE_PATH.startsWith('/'), '必须是绝对路由')
  assert.ok(DEFAULT_ROUTE_PATH.endsWith('/monitor-url'), '形状自证：以路由名结尾')
  assert.equal(BASIC_ROUTES.length, 4)
  assert.deepEqual([...BASIC_ROUTES], ['/status', '/activity', '/history', '/config'])
})

test('resolveMonitorBase：读端口文件 → http://127.0.0.1:<port>', () => {
  withTmp((dir) => {
    writeFileSync(join(dir, PORT_FILE), '52755', 'utf8')
    assert.equal(resolveMonitorBase({ root: dir, portFile: PORT_FILE }),
      'http://127.0.0.1:52755')
  })
})

test('resolveMonitorBase：每次调用现读（权威重启换端口 → 解析跟着变）', () => {
  withTmp((dir) => {
    const pf = join(dir, PORT_FILE)
    writeFileSync(pf, '56093', 'utf8')
    assert.equal(resolveMonitorBase({ root: dir, portFile: PORT_FILE }),
      'http://127.0.0.1:56093')
    writeFileSync(pf, '52755', 'utf8')   // —— 软件重启，端口漂移 ——
    assert.equal(resolveMonitorBase({ root: dir, portFile: PORT_FILE }),
      'http://127.0.0.1:52755')
  })
})

test('⭐ 文件缺失 ⇒ 抛错，**且绝不回落任何地址**', () => {
  withTmp((dir) => {
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: PORT_FILE }),
      /读不到监控端口/)
    // 反面语料：不许"偷偷回落到某个地址"——用**不含项目字面量**的判法：
    // 失败路径上任何地址都不该被造出来（造出来必然含 127.0.0.1）。
    let got = ''
    let msg = ''
    try { got = resolveMonitorBase({ root: dir, portFile: PORT_FILE }) }
    catch (err) { msg = (err as Error).message }
    assert.equal(got, '')
    assert.ok(!msg.includes('127.0.0.1'), `失败信息里不该出现地址：${msg}`)
  })
})

test('内容非数字 / 越界 ⇒ 抛错', () => {
  withTmp((dir) => {
    const pf = join(dir, PORT_FILE)
    writeFileSync(pf, 'http://127.0.0.1:9999/mcp', 'utf8')   // 旧写法（整条 URL）
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: PORT_FILE }), /内容非法/)
    writeFileSync(pf, '70000', 'utf8')
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: PORT_FILE }), /越界/)
    writeFileSync(pf, '0', 'utf8')
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: PORT_FILE }), /越界/)
  })
})

test('⭐ 端口文件名没配（空）⇒ 抛错（**库不替项目猜文件名**）', () => {
  withTmp((dir) => {
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: '' }), /未配置/)
    assert.throws(() => resolveMonitorBase({ root: dir, portFile: '   ' }), /未配置/)
  })
})

test('portFile 可给绝对路径（判据/一次性验证用）', () => {
  withTmp((dir) => {
    const pf = join(dir, 'elsewhere.port')
    writeFileSync(pf, '41000', 'utf8')
    assert.equal(resolveMonitorBase({ root: '/nonexistent', portFile: pf }),
      'http://127.0.0.1:41000')
  })
})

test('handler：解析得到 ⇒ 200 + {base}（含 no-store，面板不吃缓存）', () => {
  withTmp((dir) => {
    writeFileSync(join(dir, PORT_FILE), '41001', 'utf8')
    const res = fakeRes()
    makeMonitorUrlHandler({ root: dir, portFile: PORT_FILE })({} as any, res)
    assert.equal(res.statusCode, 200)
    assert.deepEqual(JSON.parse(res.body), { base: 'http://127.0.0.1:41001' })
    assert.equal(res.headers['Cache-Control'], 'no-store')
    assert.match(res.headers['Content-Type'], /application\/json/)
  })
})

test('handler：解析不了 ⇒ 503 + 可读错误（不是 200 空 base）', () => {
  withTmp((dir) => {
    const res = fakeRes()
    makeMonitorUrlHandler({ root: dir, portFile: PORT_FILE })({} as any, res)
    assert.equal(res.statusCode, 503)
    const body = JSON.parse(res.body)
    assert.ok(typeof body.error === 'string' && body.error.includes('监控端点未知'))
    assert.equal(body.base, undefined)      // 不许回半截地址
  })
})

test('sendJson：Content-Length 是**字节数**（中文错误体不能按字符算）', () => {
  const res = fakeRes()
  sendJson(res, 503, { error: '监控端点未知：读不到' })
  assert.equal(Number(res.headers['Content-Length']), Buffer.byteLength(res.body, 'utf8'))
  assert.notEqual(Number(res.headers['Content-Length']), res.body.length)  // 中文体两值不等
})
