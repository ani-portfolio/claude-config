import type { Limit } from '../types'

export const WIDTH = 20

export const LABELS: Record<string, string> = {
  five_hour: 'Session (5hr)',
  seven_day: 'Weekly (7 day)',
}

// Raw colours, not theme keys: the VS Code panel draws a theme key it does not know as plain text.
// The values are Claude Code's own dark theme green, yellow and red.
export const GREEN = '#4EBA65'
export const YELLOW = '#FFC107'
export const RED = '#FF6B80'

// Green below 50%, yellow from 50% to 90%, red from 90%.
export const colorFor = (percent: number) => (percent >= 90 ? RED : percent >= 50 ? YELLOW : GREEN)

export const bar = (percent: number) => {
  const filled = Math.round((Math.min(Math.max(percent, 0), 100) / 100) * WIDTH)
  return '█'.repeat(filled) + '░'.repeat(WIDTH - filled)
}

export const resetsIn = (resetsAt: string | undefined, now: number) => {
  if (!resetsAt) return ''
  const ms = Date.parse(resetsAt) - now
  if (!(ms > 0)) return 'Resets soon'
  const minutes = Math.round(ms / 60000)
  if (minutes < 60) return `Resets in ${minutes}m`
  const hours = Math.round(minutes / 60)
  if (hours < 48) return `Resets in ${hours}h`
  return `Resets in ${Math.round(hours / 24)}d`
}

export const shown = (limits: Limit[]) =>
  limits.filter(l => l.kind in LABELS).sort((a, b) => a.kind.localeCompare(b.kind))
