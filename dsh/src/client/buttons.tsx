/**
 * 会话头部「◈ 监控」按钮：开右栏并打开 ML 监控 tab。
 *
 * 自包含内联样式（无外部 CSS 依赖，随 client bundle 打进合并包）。
 * 移植自 EL `dsh/src/client/buttons.tsx`：**只保留监控按钮**——EL 的「→ GUI」
 * 按钮要 `/re0-gui`（host 半写 `.mode-request`），ML 的切回在 P2 走看门人控制台，
 * 该按钮随 `/ml-gui` 命令一起属于 P3（见 `docs/mecha/07-...md` §4）。
 */
import type { CSSProperties } from 'react'

/** ◈监控 注入面：开右栏 + 打开我们的 tab（无状态动作）。 */
export interface CockpitInjected {
  openCockpit(): void
}

export type CockpitButtonProps = { openCockpit?: () => void } & Record<string, unknown>

const BTN: CSSProperties = {
  padding: '2px 10px', borderRadius: 6, cursor: 'pointer',
  border: '1px solid rgba(128,128,128,0.35)', background: 'transparent',
  color: 'inherit', font: 'inherit', fontSize: '0.9em',
}

/** ◈监控：点击开右栏并打开 ML 监控 tab。 */
export function CockpitButton({ openCockpit }: CockpitButtonProps) {
  return (
    <button type="button" style={BTN} title="打开 ML 监控（右栏：写事件 / 配置态 / 运行记录）"
      onClick={() => openCockpit?.()}>
      ◈ 监控
    </button>
  )
}
