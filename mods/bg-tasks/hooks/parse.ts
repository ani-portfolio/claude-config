export type TaskNotice = { taskId: string; status?: string }

const BLOCK = /<task-notification>([\s\S]*?)<\/task-notification>/g
const TERMINAL = new Set(['completed', 'failed', 'killed', 'stopped', 'error', 'cancelled', 'timeout'])

const tag = (body: string, name: string) =>
  body.match(new RegExp(`<${name}>\\s*([^<]*?)\\s*</${name}>`))?.[1]

/** Every task notification in a delivered prompt; one prompt may batch several. */
export const parseNotices = (text: string): TaskNotice[] => {
  const bodies = [...text.matchAll(BLOCK)].map(m => m[1] ?? '')

  return (bodies.length > 0 ? bodies : [text]).flatMap(body => {
    const taskId = tag(body, 'task-id')
    const status = tag(body, 'status')?.toLowerCase()

    return taskId ? [{ taskId, ...(status ? { status } : {}) }] : []
  })
}

/** A monitor's event notices carry no terminal status; everything else ends on its notice. */
export const isFinished = (notice: TaskNotice, isMonitor: boolean) =>
  notice.status ? TERMINAL.has(notice.status) : !isMonitor

export const elapsed = (ms: number) => {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m}m${String(s % 60).padStart(2, '0')}s` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}m`
}

export const clip = (text: string, width: number) =>
  text.length <= width ? text : `${text.slice(0, Math.max(1, width - 1))}…`
