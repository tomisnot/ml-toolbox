import { defineConfig } from 'tsdown'

/**
 * 两个产物（对应 package.json 的 "." 与 "./client" 导出）：
 *
 * · `lib/index.mjs`  —— **HOST 半**（node 平台）。⭐ **2026-10-10 起它只是空载体**：
 *   监控**地址路由**已上提进框架（框架的 host 插件按 `MECHA_PANEL_PORT_FILES` 现取端口）
 *   ⇒ 本产物**什么都不注册**，只为"一条 patch entry 带一个包"的加载形状而存在
 *   （client 半随之进 dsh）。
 *   ⛔ **原先这里写着"不移植 EL 的自愈 MCP 桥 / 只在命中 R1 时上"** —— 2026-10-10
 *   用户裁决（方案 A）后**那座桥在全生态已删除**（资产 + EL/GP 接线），
 *   "上 P3"这个动作不再存在：工具面只有一条腿 = **框架的聚合端点**。
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
