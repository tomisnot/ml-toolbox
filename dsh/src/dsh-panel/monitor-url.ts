/**
 * 监控端点**地址**：dsh **同源只读路由**（host 半）。**参考实现**，不是框架能力。
 *
 * ## ⚠ 本模块是 **node-only**（host 半）：client 半**不许** import 它
 *
 * 它顶层要 `node:fs` / `node:path`（读端口文件、拼路径）。而 **client 半是浏览器 bundle**
 * ⇒ 一旦 client 半（或它的 import 闭包）碰到本模块，`node:fs` 就进包、**加载即失败**
 * （2026-09-26 的真实缺陷：`esbuild --bundle --platform=browser` 报
 * `Could not resolve "node:fs"`，指到本文件）。**浏览器安全的常量住 `routes.ts`**；
 * 本模块只放要 node 的实现。边界由 `monitor-client.test.ts` 的**导入闭包检查**机械守着。
 *
 * ## 它解决什么
 *
 * 面板（client 半）不能自己读磁盘；而监控端点的端口是**动态分配**的（`mecha.cockpit`
 * 起端点时 `port=0`，把实际端口写进项目根的裸端口文件）。于是"地址"必须由 host 半
 * 转一次：host 读端口文件 → 本路由回 `{base}` → 面板 fetch 同源路由拿到地址。
 *
 * ## 三条设计点（纪律取自本工程里跑得最久的那份实现）
 *
 * 1. ⭐ **绝不回落默认端口**：端口文件不存在 / 内容非数字 / 端口越界 ⇒ **抛错**，
 *    路由回 **`503` + 可读错误**。回落会把"权威没起来"显示成"**连上了但空面板**"——
 *    本工程反复在防的假绿（硬编码端口时代的真实故障模式）。
 *    ⚠ 同一条纪律**也适用于端口文件名**：所以 `portFile` 是**必填参数**，库**不提供
 *    默认文件名**（猜错名字 = 同一个病上一层）。
 * 2. **每次调用现读**：端口漂移（权威重启换端口）后**下一拍**就落在新端口上，
 *    不需要重启 dsh。本模块**不缓存**。
 * 3. **纯逻辑 + 可注入路径**：`resolveMonitorBase({root, portFile})` 是纯函数，
 *    判据可在 Node 里直接喂真文件/坏文件（见同目录 `monitor-url.test.ts`）。
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'

/** 监控端点基址的解析选项。`portFile` **必填**（无默认名）。 */
export interface MonitorUrlOptions {
  /** 项目根（权威与 dsh 的 cwd 同源）。缺省 = `process.cwd()`。 */
  root?: string
  /** 端口文件名（相对 root）或绝对路径。**必填**：空/缺 ⇒ 抛错。 */
  portFile: string
}

/** 端口文件解析失败（调用方一律转 `503`，**不回落到任何默认端口**）。 */
export class MonitorAddressError extends Error {}

function portFilePath(opts: MonitorUrlOptions): string {
  const name = (opts.portFile ?? '').trim()
  if (!name) {
    throw new MonitorAddressError(
      '端口文件名未配置（portFile 为空）：本库不替项目猜文件名——' +
        '请在参数块 PANEL_CONFIG.PORT_FILE 里填成权威实际写的那个文件',
    )
  }
  const root = opts.root && opts.root.trim() ? opts.root.trim() : process.cwd()
  return name.includes('/') || name.includes('\\') ? name : join(root, name)
}

/**
 * 解析监控端点基址（`http://127.0.0.1:<port>`）。
 *
 * 任何一步不对都**抛 `MonitorAddressError`**（调用方转成 503），
 * **不回落到任何默认端口、也不用默认文件名**。
 */
export function resolveMonitorBase(opts: MonitorUrlOptions): string {
  const path = portFilePath(opts)
  let raw = ''
  try {
    raw = readFileSync(path, 'utf8').trim()
  } catch (err) {
    throw new MonitorAddressError(
      `读不到监控端口（${path} 不存在）：软件没在 AI 模式跑？(${(err as Error).message})`,
    )
  }
  if (!/^\d+$/.test(raw)) {
    throw new MonitorAddressError(`监控端口文件内容非法：${JSON.stringify(raw)}（${path}）`)
  }
  const port = Number(raw)
  if (!(port > 0 && port <= 65535)) {
    throw new MonitorAddressError(`监控端口越界：${port}（${path}）`)
  }
  return `http://127.0.0.1:${port}`
}

/** 最小响应替身接口（host 半只需要这三件事；判据可喂假 res）。 */
export interface JsonResponder {
  statusCode: number
  setHeader(name: string, value: string): void
  end(body: string | Uint8Array): void
}

/** 写一份 JSON 响应（handler 拥有完整响应生命周期，故自己收尾）。 */
export function sendJson(res: JsonResponder, status: number, payload: unknown): void {
  const body = Buffer.from(JSON.stringify(payload), 'utf8')
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  // ⚠ **字节数**，不是字符数：中文错误体按字符算会让客户端截断（本工程踩过）。
  res.setHeader('Content-Length', String(body.length))
  res.setHeader('Cache-Control', 'no-store')   // 面板不吃缓存：地址每次现取
  res.end(body)
}

/** 路由处理：能解析就 `200 {base}`；解析不了就 `503` + 可读错误（**不回落**）。 */
export function makeMonitorUrlHandler(opts: MonitorUrlOptions) {
  return function monitorUrlHandler(_req: IncomingMessage, res: ServerResponse): void {
    try {
      sendJson(res, 200, { base: resolveMonitorBase(opts) })
    } catch (err) {
      // ⚠ 失败体里**没有** `base` 字段：不许回半截地址（否则面板会拿它去 fetch）。
      sendJson(res, 503, { error: `监控端点未知：${(err as Error).message}` })
    }
  }
}
