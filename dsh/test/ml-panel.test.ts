/**
 * **ML 项目自有**的面板判据（资产包之外的那一半）。
 *
 * ## 依赖化之后，本文件守什么、不守什么（逐条交代"承诺现在由谁守"）
 *
 * 2026-10-01：面板资产从"本仓逐字副本"改为 **`@mecha/dsh-panel` 的 `file:` 依赖**
 * （junction ⇒ 磁盘上只有一份代码，与 EL/GP 共用）。原 `panel-assets.test.ts` 随之退场，
 * 它守的三件事各有归属：
 *
 * | 原判据守的 | 现在由谁守 |
 * |---|---|
 * | 副本与参考实现**逐字一致**（10 文件指纹 + R8 空转自证） | **结构性消灭**：副本不存在了（路径是 junction ⇒ 改它等于改 mecha）。资产自带的 4 个共享单测由**框架门禁**（`mecha/checks/dsh_panel_selfcheck.py`）跑 |
 * | "参数块确实填了 ML 的值"（正则钉字面量） | **本文件第 1 条**：改成"注入后**读回来**等于 ML 的值"（比钉字面量强：它验的是注入真的在链上，而不只是文件里有那个字符串） |
 * | ⭐ client 闭包**零 `node:`** + 裸包白名单 | **本文件第 2 条**（**结构性无法消灭**，故保留并重写：从"扫仓内相对路径"改成"**跟着 junction 走进资产包**"，见下） |
 *
 * ## 第 2 条为什么必须跟着 junction 走
 *
 * 闭包检查**穿过 `node_modules/@mecha/dsh-panel/`**（junction 指向 mecha 仓那份）⇒ 一条检查
 * 同时证明两件事：**资产真的被解析到**，且**它的 client 侧没把 node-only 模块拖进浏览器
 * bundle**。这是 EL 先例（`energy-level/dsh/test/re0-panel.test.ts`）的口径。
 *
 * ⚠ **断言意图，不写死资产内部目录布局**：资产按入口分过家（`panel/`、`host/`），将来还可能再动
 * ⇒ 本文件只断言"进了 `pkg/` 且以 `/routes.ts` 结尾"这类**意图级**事实，不写 `panel/routes.ts`。
 *
 * 跑：`npm test`（与仓内其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { configurePanel, panelConfig } from '@mecha/dsh-panel/panel/panel-config.ts'
import { makeMonitorUrlHandler } from '@mecha/dsh-panel/host/monitor-url.ts'
import { ML_PANEL } from '../src/ml-panel-params.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
/**
 * 资产包的落点。`file:` 依赖 ⇒ `node_modules/@mecha/dsh-panel` 是指向 mecha 仓的 **junction**。
 *
 * ⚠ 这里用 `realpathSync` **判等**（而不是拿 junction 路径去比字符串）：静态遍历按 junction 路径
 * 走（`spec.startsWith(PKG)` 后 `join(PANEL_DIR, …)`），而 `label()` 要判"这个文件是不是资产里的"
 * ⇒ 两边都归一化到真实路径才不会因"经不经 junction"而假红。
 *
 * ⚠⚠ **不要给 `npm test` 加 `--preserve-symlinks`**（实测踩过，代价是两条判据全红）：
 * `--preserve-symlinks` 会让模块标识**停在 `node_modules/…` 路径上**，而 Node 24 的原生类型剥离
 * **拒绝 node_modules 下的任何 `.ts`**（`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`）
 * ⇒ 资产包（全是 `.ts`）**根本 import 不进来**。不带该旗标时 Node 按 realpath 规范化，
 * 资产在 node_modules 之外 ⇒ 类型剥离正常（对照实验：不带→`function`，带→抛错）。
 * 这是 node 运行期的事，与 `tsconfig.preserveSymlinks`（**类型检查期**必须开）互不冲突。
 */
const PANEL_DIR = realpathSync(join(HERE, '..', 'node_modules', '@mecha', 'dsh-panel'))
/** 资产包的**裸包名前缀**（两半都经它 import）。 */
const PKG = '@mecha/dsh-panel/'

const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/** 路由字面量在参数家之外出现 ⇒ 单一来源破了（第二处字面量就是漂移的温床）。 */
function literalOutsideParams(literal: string, files: string[]): string[] {
  return files.filter((f) => read(f).includes(literal))
}

// ---------------------------------------------------------------- 1 注入往返
test('⭐ ML 注入的面板参数：值就是 ML 的，且注入能被读回（往返）；未注入即抛', () => {
  // ⚠ 为什么这条**替代**了原来"正则钉参数块字面量"那两条断言：漏注入**类型检查不会报错**
  // （`src/index.ts` 只在 `apply()` 里才读 routePath、client 半只在 `title()` 里读 title）
  // ⇒ 必须有一条不依赖浏览器的判据把"注入真的在链上"钉住。
  assert.equal(ML_PANEL.routePath, '/ml-toolbox/monitor-url')
  assert.equal(ML_PANEL.portFile, '.ml-monitor-port')
  assert.ok(ML_PANEL.title.trim().length > 0, '标题不许空')

  const dispose = configurePanel(ML_PANEL)
  try {
    assert.deepEqual(panelConfig(), ML_PANEL, '注入后必须原样读回（注入点真的在链上）')
  } finally {
    dispose()
  }
  // 对偶（不许空洞）：disposer 清掉之后，读它必须**响亮失败**——
  // 资产刻意不给中性默认（否则"忘了注入"会演成"面板连上了但是空的"）。
  assert.throws(() => panelConfig(), /未注入/, '未注入时必须抛，不许给默认值')
})

// ---------------------------------------------------------------- 2 浏览器安全（结构）
/**
 * 沿**相对 import 与 `@mecha/dsh-panel/…`** 走一遍导入闭包（node 侧、不启浏览器）。
 *
 * ⚠ 闭包**跟着 junction 走进资产包**；其它裸包名（`react` 等）由 dsh 宿主经模块表提供，不追。
 * 标签用 `pkg/` / `ml/` 前缀，**断言不写死资产内部目录名**。
 */
function closureOf(entryAbs: string): { seen: string[]; nodeHits: string[]; bareHits: string[] } {
  const strip = (src: string) => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1')
  /**
   * 把 specifier 解析成磁盘上的真文件。
   *
   * ⚠ 必须容错两种写法：裸包/相对路径都**可能不带扩展名**（本仓 `./buttons` 就不带，
   * 资产内部一律带 `.ts`）。不解析扩展名会在遍历里 ENOENT（实测踩过）。
   */
  const resolve = (base: string): string | undefined => {
    for (const cand of [base, base + '.ts', base + '.tsx', base + '/index.ts', base + '/index.tsx']) {
      try {
        readFileSync(cand)
        return cand
      } catch {
        /* 试下一个 */
      }
    }
    return undefined
  }
  const label = (p: string): string => {
    // 两边都取真实路径再判等：遍历按 junction 路径走，而"是不是资产里的"要按真实落点算。
    const real = realpathSync(p)
    return real.startsWith(PANEL_DIR)
      ? 'pkg/' + relative(PANEL_DIR, real).replace(/\\/g, '/')
      : 'ml/' + relative(realpathSync(SRC), real).replace(/\\/g, '/')
  }
  const seen = new Set<string>()
  const nodeHits: string[] = []
  const bareHits: string[] = []
  const queue = [entryAbs]
  while (queue.length) {
    const file = queue.shift() as string
    if (seen.has(file)) continue
    seen.add(file)
    for (const line of strip(readFileSync(file, 'utf8')).split('\n')) {
      if (/^\s*import\s+type\b/.test(line)) continue   // 类型导入会被剥掉，不是运行时依赖
      const m = line.match(/from\s+'([^']+)'/)
      if (!m) continue
      const spec = m[1] ?? ''
      if (spec.startsWith('node:')) { nodeHits.push(`${label(file)} → ${spec}`); continue }
      let next: string | undefined
      if (spec.startsWith(PKG)) next = resolve(join(PANEL_DIR, spec.slice(PKG.length)))
      else if (spec.startsWith('.')) next = resolve(join(dirname(file), spec))
      else {
        // 其它裸包名：dsh 宿主经模块表注入，不追——但**也不许**随便出现（见下）。
        if (spec !== 'react' && spec !== 'react/jsx-runtime' && !spec.startsWith('@deepseek-ai/')) {
          bareHits.push(`${label(file)} → ${spec}`)
        }
        continue
      }
      if (next === undefined) bareHits.push(`${label(file)} → ${spec}（解析不到）`)
      else queue.push(next)
    }
  }
  return { seen: [...seen].map(label), nodeHits, bareHits }
}

test('⭐ ML 入口 + 资产包的 import 闭包里没有 node:*，且真的走进了资产（跟着 junction）', () => {
  const { seen, nodeHits, bareHits } = closureOf(join(SRC, 'client', 'index.ts'))
  assert.deepEqual(nodeHits, [],
    `闭包里出现 node 内建：${nodeHits.join('、')}`
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在资产的 host/monitor-url.ts）')
  assert.deepEqual(bareHits, [],
    `闭包里出现未允许的裸包：${bareHits.join('、')}`
    + '（只许 react / react-jsx-runtime 与 @deepseek-ai/* —— 它们由 dsh 模块表注入）')
  // R8 自证：闭包必须**真的走过了若干文件**，且**真的进了资产包**（不写死资产内部目录名）
  assert.ok(seen.length >= 5, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/routes.ts')),
    `闭包必须走进资产包（junction 没被跟随？）：${seen}`)
  assert.ok(seen.includes('ml/ml-panel-params.ts'),
    `闭包里应当有 ML 的参数家（否则参数没接上）：${seen}`)
  assert.ok(!seen.some(s => s.includes('monitor-url')),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：资产的 node 半**必须**仍依赖 node —— 否则上一条"没东西可抓"
  assert.match(readFileSync(join(PANEL_DIR, 'host', 'monitor-url.ts'), 'utf8'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 3 单一来源
test('⭐ 单一来源：路由字面量只在 ml-panel-params.ts，两半各注入一次', () => {
  // host 半：自己注入（**不换 `mountHostPlugin` 骨架**——ML 没有 MCP 桥，骨架主要装桥的生命周期）
  assert.match(read('index.ts'), /^configurePanel\(ML_PANEL\)$/m,
    'host 半必须在模块顶层注入 ML 的面板参数')
  // client 半：另一份 bundle / 另一个进程 ⇒ 它自己把参数交出去（host 那次到不了浏览器）
  // ⚠ 2026-10 阶段①：**交出方式**从"顶层 `configurePanel(ML_PANEL)`"变成
  //   "`mountClientPanel(..., { panelConfig: ML_PANEL, ... })`"（按钮/ tab 注册归框架）。
  //   ⇒ 判据跟着改**机制**，但断言的**性质**没变：**client 半必须把 ML 那份参数交进去**
  //   （资产的 `openPanelConfig` 未注入即抛 ⇒ 少了它就是浏览器里才炸的那类故障）。
  assert.match(read('client/index.ts'), /panelConfig:\s*ML_PANEL/,
    'client 半必须把 ML 的面板参数交给框架 helper（另一份 bundle，host 那次到不了这里）')
  assert.match(read('ml-panel-params.ts'), /'\/ml-toolbox\/monitor-url'/,
    '路由字面量必须住在 ml-panel-params.ts（ML 的参数家）')
  const offenders = literalOutsideParams(ML_PANEL.routePath,
    ['index.ts', 'client/index.ts', 'ml-panel-params.ts'])
  assert.deepEqual(offenders.filter((f) => f !== 'ml-panel-params.ts'), [],
    `路由字面量被抄进了参数家之外：${offenders.join('、')}`)
})

// ---------------------------------------------------------------- 4 参数真的被资产消费
/**
 * ⭐ 迁移验收（照 EL 的口径）：**注入的参数真的被资产消费** —— 拿真 handler、喂真端口文件，
 * 断言 200/503 与**端口值**都对。这不启 dsh（用户明令），只走资产的数据层。
 */
test('⭐ 注入的 portFile 真的被资产消费：有端口文件→200，缺→503（不回落）', () => {
  const root = mkdtempSync(join(tmpdir(), 'ml-panel-'))
  const fake = () => {
    const captured: { status: number; body: unknown } = { status: 0, body: null }
    const res = {
      statusCode: 0,
      setHeader() { /* 不关心头 */ },
      end(body: string | Uint8Array) {
        captured.status = res.statusCode
        captured.body = JSON.parse(String(body))
      },
    }
    return { res, captured }
  }
  try {
    const dispose = configurePanel(ML_PANEL)
    try {
      // 用**注入后的** portFile（不是字面量）建 handler —— 这一步就证明"注入 → 资产读取"在链上
      const handler = makeMonitorUrlHandler({
        get root() { return root },
        get portFile() { return panelConfig().portFile },
      })
      // ① 有端口文件 ⇒ 200 + 正确 base
      writeFileSync(join(root, ML_PANEL.portFile), '54321\n', 'utf8')
      const ok = fake()
      handler({} as never, ok.res as never)
      assert.equal(ok.captured.status, 200, `应当是 200：${JSON.stringify(ok.captured)}`)
      assert.deepEqual(ok.captured.body, { base: 'http://127.0.0.1:54321' })
      // ② 缺端口文件 ⇒ **503 + 可读错误，且没有 base**（"没起来"不许演成"连上了但空的"）
      rmSync(join(root, ML_PANEL.portFile))
      const bad = fake()
      handler({} as never, bad.res as never)
      assert.equal(bad.captured.status, 503, `应当是 503：${JSON.stringify(bad.captured)}`)
      assert.ok(!('base' in (bad.captured.body as object)), '503 体里不许有 base')
      assert.match(JSON.stringify(bad.captured.body), /监控端点未知/)
    } finally {
      dispose()
    }
    // ③ 反向：**未注入**时 handler 也必须 503（而不是抛穿/回默认端口）——这正是 getter 存在的理由
    const noInject = fake()
    makeMonitorUrlHandler({ get root() { return root } , get portFile() { return panelConfig().portFile } })(
      {} as never, noInject.res as never)
    assert.equal(noInject.captured.status, 503, '未注入时必须 503（响亮），不许抛穿或回默认')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
