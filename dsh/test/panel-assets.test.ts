/**
 * ML 侧守卫：`src/dsh-panel/` 的副本**与参考实现逐字一致**（除参数块）。
 *
 * ## 为什么是"常量指纹"，不是"运行时读上游"
 *
 * 判据**只读本仓的文件**：把参考实现提交态正文的 `sha256` 写成下面的常量，
 * `REFERENCE_REVISION` **只作出处说明**。⇒ 零跨仓读取、零 git（本工程两次教训：
 * gitignore 的文件逃出 `git diff`；目录 junction 让两条路径其实是同一个文件）。
 * 上游真发新版时：**整份重新复制 + 同步更新常量**，不是就地打补丁。
 *
 * ## 比的是"归一化行尾后的正文"
 *
 * `core.autocrlf` 会让同一内容在不同检出状态下字节不同 ⇒ 先把 `\r\n` 归一成 `\n`
 * 再取指纹。裸字节比对会被行尾配置打败。
 *
 * ## 指纹表收哪些
 *
 * 只收**行为载体**（代码 + 单测）。`panel-config.ts` 是参数块（**必然**不同）、
 * `README.md` 是 prose（就地批注不改行为）⇒ 两者都**不进表**，由另一条判据单独钉。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** 出处说明（**只作出处**；比对靠 `CARRIER_SHA256`，判据运行时**不读 mecha**）。 */
export const REFERENCE_REVISION = 'ab28452:mecha/dsh-panel'

/** 行为载体 = 参考实现的代码 + 单测（10 个文件），LF 归一后的 sha256。 */
export const CARRIER_SHA256: Record<string, string> = {
  'routes.ts': 'e2b5bc35d84d368548146a4c7d6c4a2ef733b3191480f6a9d0131045c1295963',
  'monitor-url.ts': '9808f71a7179857f56baa25652cfc77e2d109f3d92d643aaf2c3ebc42f95c2a3',
  'monitor-client.ts': '200a7764f2d0139c17dd91294d02dd165ad0f51ad782ad39b3bb67708153e93c',
  'MonitorTabBody.tsx': '8ed1233f18e32a9c4b6af37ebd69937fe63f5522ce2a410bb0dab8ec9339d3d6',
  'panel-data.ts': 'cc32f9c7692fdbf2f0495d23f2f4bab2ac3e27620795e92f734132c96a0153bc',
  'panel-view.ts': '560103c18231df5bad3d18a77f16f0ed6cceab513928ef765dbdb2afec9a1aca',
  'monitor-url.test.ts': '90db86e3b5217b89b18b082f37c502ef6750c54b2de58ad202fb71eecd9cfd29',
  'monitor-client.test.ts': '8dcff14f8fe5fb2f505ba7e9e845adab4f8c28c4af87c0234856817cecea4487',
  'panel-data.test.ts': 'b3ece6c3f436f22b5abd3f7ffb295d3799f0fadf2628d4d791312bbd4445152a',
  'panel-view.test.ts': 'bd2afe86800130e9ff8c03081a5696bd2c9f4c23fe71070c3fca57c72931d1b4',
}

const PANEL_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'dsh-panel')

/** 行尾归一（`\r\n` → `\n`）：裸字节会被 `core.autocrlf` 打败。 */
export const normalizeEol = (s: string): string => s.replace(/\r\n/g, '\n')

/** 正文指纹（LF 归一后 UTF-8 的 sha256）。 */
export const contentDigest = (s: string): string =>
  createHash('sha256').update(normalizeEol(s), 'utf8').digest('hex')

function readPanel(name: string): string {
  return readFileSync(join(PANEL_DIR, name), 'utf8')
}

test('⭐ 除参数块外的正文与参考实现逐字一致（常量指纹：零跨仓、零 git）', () => {
  for (const [name, want] of Object.entries(CARRIER_SHA256)) {
    const got = contentDigest(readPanel(name))
    assert.equal(
      got, want,
      `${name} 与参考实现（${REFERENCE_REVISION}）不再逐字一致 ⇒ 要么是就地改了副本` +
      '（不许：只许改 panel-config.ts），要么是重新复制后忘了同步指纹常量。' +
      `\n  期望 ${want}\n  实际 ${got}`)
  }
  // R8 自证：真的比过 10 个"行为载体"，不是空转
  assert.equal(Object.keys(CARRIER_SHA256).length, 10)
})

test('⭐ 参数块与 README 在指纹表之外，且参数块确实填了 ML 的值', () => {
  assert.ok(!('panel-config.ts' in CARRIER_SHA256),
    '参数块**必然**与参考实现不同 ⇒ 不该进指纹表（进了就是自找假红）')
  assert.ok(!('README.md' in CARRIER_SHA256),
    'prose 不进指纹表（就地批注不改行为，钉它只制造噪声）')
  const cfg = readPanel('panel-config.ts')
  // 填漏了 PORT_FILE ⇒ 路由一律 503（资产刻意如此），所以这里必须钉住实际值
  assert.match(cfg, /ROUTE_PATH: '\/ml-toolbox\/monitor-url'/, 'ROUTE_PATH 未填成 ML 的')
  assert.match(cfg, /PORT_FILE: '\.ml-monitor-port'/, 'PORT_FILE 未填成 ML 的端口文件名')
  // R17：参考实现那行 `import { DEFAULT_ROUTE_PATH }` 在填了字面量后必须删掉
  assert.ok(!/^import .*DEFAULT_ROUTE_PATH/m.test(cfg),
    'panel-config.ts 仍 import DEFAULT_ROUTE_PATH ⇒ 那是"声明了却不被读"的死缝（R17）')
})

test('R7 红证：指纹函数对**单字符**扰动敏感（不是哑守卫）', () => {
  const src = readPanel('routes.ts')
  const base = contentDigest(src)
  assert.notEqual(contentDigest(src + '\n'), base, '末尾加一个换行竟不影响指纹 ⇒ 守卫是哑的')
  assert.notEqual(contentDigest(src.replace('mecha', 'mechb')), base,
    '改一个字符竟不影响指纹 ⇒ 守卫是哑的')
  // 对偶：行尾归一后，CRLF 版本必须与 LF 版本**同指纹**（否则 Windows 检出会假红）
  assert.equal(contentDigest(src.replace(/\n/g, '\r\n')), base,
    '行尾归一失效：CRLF 与 LF 得出了不同指纹')
})

// ---------------------------------------------------------------- 入口级闭包守卫

/** 本仓 client 半的**入口**（不是资产目录内部）：闭包从这里开始走。 */
const CLIENT_ENTRY = 'src/client/index.ts'

/** 唯一允许的裸包名：dsh 宿主经模块表注入的 react（项目的 bundler 把它外置）。 */
const ALLOWED_BARE = new Set(['react', 'react/jsx-runtime'])

const DSH_DIR = join(dirname(fileURLToPath(import.meta.url)), '..')

/** 把相对 specifier 解析成仓内相对路径（`./x`、`../y`、带或不带扩展名）。 */
function resolveSpec(fromRel: string, spec: string): string | undefined {
  const base = join(dirname(fromRel), spec).replace(/\\/g, '/')
  for (const cand of [base, base + '.ts', base + '.tsx', base + '/index.ts',
                      base + '/index.tsx']) {
    try {
      readFileSync(join(DSH_DIR, cand))
      return cand
    } catch {
      /* 试下一个 */
    }
  }
  return undefined
}

/** 剥注释（否则文档里"举例说明"的 `from 'node:fs'` 会被误判成真 import）。 */
const stripComments = (src: string): string => src
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/(^|[^:])\/\/.*$/gm, '$1')

test('⭐ 从**本仓入口**出发：client 半闭包零 `node:`，裸包只许 react（照 GP 那条）', () => {
  const seen = new Set<string>()
  const nodeHits: string[] = []
  const bareHits: string[] = []
  const queue = [CLIENT_ENTRY]
  while (queue.length) {
    const rel = queue.shift() as string
    if (seen.has(rel)) continue
    seen.add(rel)
    const src = stripComments(readFileSync(join(DSH_DIR, rel), 'utf8'))
    for (const line of src.split('\n')) {
      if (/^\s*import\s+type\b/.test(line)) continue   // 类型导入被剥掉，不是运行时依赖
      const m = line.match(/from\s+'([^']+)'/)
      if (!m) continue
      const spec = m[1] ?? ''
      if (spec.startsWith('node:')) { nodeHits.push(`${rel} → ${spec}`); continue }
      if (spec.startsWith('.')) {
        const next = resolveSpec(rel, spec)
        if (next) queue.push(next)
        continue
      }
      if (!ALLOWED_BARE.has(spec)) bareHits.push(`${rel} → ${spec}`)
    }
  }
  assert.deepEqual(nodeHits, [],
    `client 半闭包里出现了 node 内建：${nodeHits.join('、')} ⇒ 浏览器 bundle 会失败`
    + '（node-only 的实现必须留在 monitor-url.ts）')
  assert.deepEqual(bareHits, [],
    `client 半闭包里出现了未允许的裸包：${bareHits.join('、')}`
    + `（只许 ${[...ALLOWED_BARE].join(' / ')}——它们是 dsh 模块表注入的）`)
  // R8 自证：闭包必须**真的走过了几个文件**，且**真的走到了共享面板**——
  // 否则"没命中"只是因为入口压根没接线（那正是 R17 禁止的假锚点）。
  assert.ok(seen.size >= 4, `闭包只走了 ${seen.size} 个文件 ⇒ 检查可能没生效：${[...seen]}`)
  assert.ok(seen.has('src/dsh-panel/monitor-client.ts'),
    `闭包里应当有共享取数管线：${[...seen]}`)
  assert.ok(seen.has('src/dsh-panel/MonitorTabBody.tsx'),
    `闭包里应当有共享面板本体（否则面板没接上，这条检查是空转）：${[...seen]}`)
  // 对偶（不许误报）：host 半**必须**仍依赖 node —— 同一条边界的两侧都要钉住
  const hostHalf = readFileSync(join(DSH_DIR, 'src/index.ts'), 'utf8')
  assert.match(hostHalf, /from '\.\/dsh-panel\/monitor-url\.ts'/,
    'host 半应当用共享的地址实现（monitor-url.ts 是 node-only 的那一半）')
})

