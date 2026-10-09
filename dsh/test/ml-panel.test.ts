/**
 * **ML 项目自有**的面板判据（资产包之外的那一半）。
 *
 * ## 2026-10-09：监控面**上提进框架**之后，本文件守的东西（逐条交代"承诺现在由谁守"）
 *
 * | 原来守的 | 现在由谁守 |
 * |---|---|
 * | "ML 注入的面板参数能读回"（往返） | ⭐ **结构性消灭**：本仓**不再有**面板参数（`ml-panel-params.ts` 已删）⇒ 没有可注入、也没有可漂的第二份。参数的家 = `ml_mecha/mecha_site.py` 的 `Service.monitor` |
 * | "声明与参数家同值"（两处对拍） | ⭐ mecha 的 `phase13_panel_decl_selfcheck`（**夹具 ↔ 三家真声明对拍**）—— 它读的就是**真声明** |
 * | "注入的 portFile 真的被资产消费"（handler 200/503） | ⭐ **框架的 host 插件**：`mecha/dsh-panel/host-plugin/mecha-panel-host.test.ts`（含**与资产实现的等价判据**） |
 * | "路由字面量只在参数家" | ⭐ **结构**：本仓不再出现那个字面量（下面第 2 条**反向**钉住"不许回来"） |
 * | client 闭包**零 `node:`** + 裸包白名单 | **本文件第 1 条**（**结构性无法消灭**：浏览器的加载面） |
 *
 * ⚠ **反向断言**（R13：删了机制要留"不许回来"的判据）：本仓**不许**再自己注册
 * 监控地址路由 / 不许再有 `ML_PANEL` / 不许再传 `panelConfig`。⚠ 同一条 `routePath`
 * 注册两次是**接线错误**（谁生效取决于顺序）—— 那是静默的错，正是要防的形态。
 *
 * 跑：`npm test`（与仓内其余单测一起）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync, realpathSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const SRC = join(HERE, '..', 'src')
const REPO = join(HERE, '..', '..')
/**
 * 资产包的落点（`file:` 依赖 ⇒ `node_modules/@mecha/dsh-panel` 是**junction**）。
 *
 * ⚠⚠ **不要给 `npm test` 加 `--preserve-symlinks`**（实测踩过）：它让模块标识停在
 * `node_modules/…`，而 Node 的原生类型剥离**拒绝处理 node_modules 下的 `.ts`**
 * （`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`）⇒ 资产包根本 import 不进来。
 */
const PANEL_DIR = realpathSync(join(HERE, '..', 'node_modules', '@mecha', 'dsh-panel'))
/** 资产包的**裸包名前缀**（两半都经它 import）。 */
const PKG = '@mecha/dsh-panel/'

const read = (rel: string): string => readFileSync(join(SRC, rel), 'utf8')

/** 去掉注释：本文件几处"不许出现 X"的断言看的是**代码**，而注释里解释
 *  "X 已删/不许回来"是**该留的**（否则下一个人查不到为什么不能走老路）。 */
const stripComments = (src: string): string => src
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/(^|[^:])\/\/.*$/gm, '$1')

/** 本仓 dsh 的源码文件（**含 `.tsx`**：死文件曾经就是这么漏掉的）。 */
function sourceFiles(): string[] {
  const out: string[] = []
  const walk = (dir: string): void => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name)
      if (e.isDirectory()) walk(p)
      else if (/\.tsx?$/.test(e.name)) out.push(relative(SRC, p).replace(/\\/g, '/'))
    }
  }
  walk(SRC)
  return out
}

function codeHits(needle: string): string[] {
  return sourceFiles().filter((f) => stripComments(read(f)).includes(needle))
}

// ---------------------------------------------------------------- 1 浏览器安全（结构）
/**
 * 沿**相对 import 与 `@mecha/dsh-panel/…`** 走一遍导入闭包（node 侧、不启浏览器）。
 *
 * ⚠ 闭包**跟着 junction 走进资产包**；其它裸包名（`react` 等）由 dsh 宿主经模块表提供，不追。
 * 标签用 `pkg/` / `ml/` 前缀，**断言不写死资产内部目录名**。
 */
function closureOf(entryAbs: string): { seen: string[]; nodeHits: string[]; bareHits: string[] } {
  const strip = stripComments
  /** 把 specifier 解析成磁盘上的真文件（裸包/相对路径都可能不带扩展名）。 */
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
    + ' ⇒ 浏览器 bundle 会失败（node-only 的实现必须留在资产的 host 面）')
  assert.deepEqual(bareHits, [],
    `闭包里出现未允许的裸包：${bareHits.join('、')}`
    + '（只许 react / react-jsx-runtime 与 @deepseek-ai/* —— 它们由 dsh 模块表注入）')
  // R8 自证：闭包必须**真的走过了若干文件**，且**真的进了资产包**（不写死资产内部目录名）
  assert.ok(seen.length >= 5, `闭包只走了 ${seen.length} 个文件 ⇒ 检查可能没生效：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/routes.ts')),
    `闭包必须走进资产包（junction 没被跟随？）：${seen}`)
  // ⚠ client 半还带着 ML **自己的**东西：呈现细化（附加页的主指标取值函数）
  assert.ok(seen.includes('ml/client/extraPages.ts'),
    `闭包里应当有 ML 的呈现细化（否则 client 半只剩一行 bootstrap）：${seen}`)
  assert.ok(seen.some(s => s.startsWith('pkg/') && s.endsWith('/panels.ts')),
    `闭包必须走进框架那份 client 插件（面板参数靠它按声明挂 tab）：${seen}`)
  assert.ok(!seen.some(s => s.includes('monitor-url')),
    `client 半不许碰到 node-only 的 monitor-url.ts：${seen}`)
  // 对偶（不许空洞）：资产的 node 半**必须**仍依赖 node —— 否则上一条"没东西可抓"
  assert.match(readFileSync(join(PANEL_DIR, 'host', 'monitor-url.ts'), 'utf8'), /from 'node:fs'/,
    'node 半应当 import node:fs（否则本检查没有可抓之物）')
})

// ---------------------------------------------------------------- 2 不许再自己注册（反向）
test('⭐ 监控面归框架 ⇒ 本仓不许再有地址路由 / ML_PANEL / panelConfig / 自己的 tab 注册', () => {
  assert.match(read('client/index.ts'),
    /export \{ apply, name, inject \} from '@mecha\/dsh-panel\/panel\/auto\.ts'/,
    'client 半必须有那一行 bootstrap（少了它 = 右栏一个 tab 都没有，且不报错）')
  for (const needle of ['ML_PANEL', 'makeMonitorUrlHandler', 'configurePanel', 'panelConfig']) {
    const hits = codeHits(needle)
    assert.deepEqual(hits, [], `${needle} 又回来了（路由/参数的第二处真值）：${hits.join('、')}`)
  }
  // ⚠ 只禁"**自己注册**"；`inject` 里声明 `sidebarRightTabs` 是**必须的**（cordis 的 ctx 是代理：
  //   框架那份插件要经本仓的 ctx 用这些服务，不声明就抛）—— 把"声明服务"当"自己注册"来禁
  //   会造出一条**假判据**（GP 侧实测踩到）。
  assert.doesNotMatch(stripComments(read('client/index.ts')),
    /sidebarRightTabs\s*\.\s*register|slots\s*\.\s*register/,
    '本仓不许再自己注册右栏 tab / 槽位（监控面板归框架）')
  // host 半**只剩载体**（dsh 的加载形状要求它存在）：不许注册路由、也不许依赖 webServer
  assert.match(stripComments(read('index.ts')), /export const inject: string\[\] = \[\]/,
    'host 半不该再依赖任何服务（地址路由归框架）')
})

// ---------------------------------------------------------------- 3 那个唯一家还在
test('⭐ 那个唯一家还在：`ml_mecha/mecha_site.py` 里有 monitor=MonitorSpec(...)', () => {
  // ⚠ 本判据**只**断言"声明还在"；**值**由 mecha 的 `phase13_panel_decl_selfcheck`
  //   拿**真声明**对拍（夹具↔真声明），不在这里抄一遍（抄一遍就是第二处真值）。
  const decl = readFileSync(join(REPO, 'ml_mecha', 'mecha_site.py'), 'utf8')
  assert.match(decl, /monitor=MonitorSpec\(/, '声明里没有 monitor=MonitorSpec(...)')
  assert.match(decl, /route_path="/, '声明里 monitor 缺 route_path（框架按它注册地址路由）')
  assert.match(decl, /port_file="/, '声明里 monitor 缺 port_file（框架按它现取端口）')
})
