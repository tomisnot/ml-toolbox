import { defineConfig } from 'tsdown'

/**
 * 两个产物（对应 package.json 的 "." 与 "./client" 导出）：
 *
 * · `lib/index.mjs`  —— **HOST 半**（node 平台）。本插件只干一件事：注册
 *   `/ml-monitor-url` command，把 ML 只读监控端点的地址告诉 client 半。
 *   **不移植** EL 的自愈 MCP 桥（`dsh/src/host/mcp-bridge.ts`）：ML 的工具接入
 *   走 dsh 内置 `@deepseek-ai/dsh-mcp-client`（方案 A）。自愈桥只在命中 R1
 *   （streamable-http "HTTP 世代"死区）时才上，触发判据见
 *   内部记录（未随仓发布，存档在仓外） §4 P3。
 *
 * · `lib/client.js`  —— **CLIENT 半**（browser 平台）。dsh 的 web client 把所有
 *   插件的 client.js 合成一个 **classic-script 合并包**，每个模块必须自注册到
 *   `window.__ModuleLoader__.load({id, factory:(require)=>{...}})`（见 dsh
 *   `packages/client/tsdown.client.ts` 的 clientConfig）。普通 ESM（顶层
 *   import/export）在合并包里是 SyntaxError → 整包崩。故这里精确复刻该格式。
 *
 * react / react/jsx-runtime 是 dsh 的 PLATFORM_MODULES（模块表提供）→ 外置，经
 * 注入的 require 解析，共享 dsh 的 React 实例（避免双 React）。
 *
 * ⚠ banner 里的 id 必须等于 package.json 的 `name`。
 */
const dshExternal = [/^@deepseek-ai\//]

export default defineConfig([
  {
    entry: { index: 'src/index.ts' },
    outDir: 'lib',
    format: 'esm',
    platform: 'node',
    target: 'node20',
    external: dshExternal,
    dts: false,
    sourcemap: true,
    clean: true,
  },
  {
    entry: { client: 'src/client/index.ts' },
    outDir: 'lib',
    format: 'cjs',
    platform: 'browser',
    target: 'es2022',
    external: ['react', 'react/jsx-runtime', ...dshExternal],
    dts: false,
    sourcemap: true,
    clean: false, // 别清掉上一步的 host 产物
    outputOptions: {
      entryFileNames: 'client.js',
      banner: 'window.__ModuleLoader__.load({ id: "ml-toolbox-dsh", factory: (require) => {',
      intro: 'var module = { exports: {} }; var exports = module.exports;',
      footer: 'return module.exports; } });',
    },
  },
])
