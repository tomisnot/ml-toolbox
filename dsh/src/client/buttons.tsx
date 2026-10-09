/**
 * 会话头部「◈ 监控」按钮：开右栏并打开 ML 监控 tab。
 *
 * 自包含内联样式（无外部 CSS 依赖，随 client bundle 打进合并包）。
 * 移植自 EL `dsh/src/client/buttons.tsx`：**只保留监控按钮**。
 * ⚠ **ML 不需要「→ GUI」按钮**：三层架构下"切到手动"是**切写权**（`Mode`），
 * 入口在**启动面板/常驻控制台**上，不再是一个往文件里写意图的 dsh 按钮
 * （旧 `/ml-gui` + `.mode-request` 那套随旧架构一起作废）。
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
