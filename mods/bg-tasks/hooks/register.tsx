import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { BgKind, BgTask } from '../types'
import { clip, elapsed, isFinished, parseNotices } from './parse'

const tasks = atom({ plugin: 'bg-tasks', key: 'tasks' } as const, {})
const agents = atom({ plugin: 'bg-tasks', key: 'agents' } as const, [])
const now = atom({ plugin: 'bg-tasks', key: 'now' } as const, 0)
const isHidden = atom({ plugin: 'bg-tasks', key: 'isHidden' } as const, false)

// A raw colour, not a theme key, which the VS Code panel draws as plain text. The same green as usage-bar.
const GREEN = '#4EBA65'

const LIVE = new Set(['pending', 'running', 'waiting'])
const POLL_MS = 2000
const SHOWN = 5
const ICON: Record<BgKind, string> = { agent: '◆', shell: '$', monitor: '◉', workflow: '⧉', remote: '☁' }
const NOUN: Record<BgKind, [string, string]> = {
  agent: ['agent', 'agents'],
  shell: ['shell', 'shells'],
  monitor: ['monitor', 'monitors'],
  workflow: ['workflow', 'workflows'],
  remote: ['remote', 'remotes'],
}

const all = async ($: EngineInterface) => [...(await read($, agents)), ...Object.values(await read($, tasks))]

const breakdown = (list: BgTask[]) =>
  (Object.keys(NOUN) as BgKind[])
    .map(kind => [kind, list.filter(t => t.kind === kind).length] as const)
    .filter(([, n]) => n > 0)
    .map(([kind, n]) => `${n} ${NOUN[kind][n === 1 ? 0 : 1]}`)
    .join(' · ')

const add = async ($: EngineInterface, task: Omit<BgTask, 'startedAt'>) => {
  const startedAt = await $.clock.now()
  await update($, tasks, map => ({ ...map, [task.id]: { ...task, startedAt } }))
}

const remove = async ($: EngineInterface, ids: string[]) => {
  const map = await read($, tasks)
  const gone = ids.flatMap(id => map[id] ?? [])
  if (gone.length === 0) return
  await update($, tasks, m => Object.fromEntries(Object.entries(m).filter(([id]) => !ids.includes(id))))
  const left = (await all($)).length
  for (const t of gone) $.ui.toast(`${NOUN[t.kind][0]} finished: ${clip(t.label, 40)} (${left} left)`)
}

/** Agents come from the engine's own list; only their start times are ours. */
const pollAgents = async ($: EngineInterface) => {
  const t = await $.clock.now()
  const before = await read($, agents)
  const live = (await $.agent.list()).filter(a => LIVE.has(a.status))
  const next: BgTask[] = live.map(a => ({
    id: a.id,
    kind: 'agent',
    label: a.description || a.name || a.type,
    startedAt: before.find(b => b.id === a.id)?.startedAt ?? t,
  }))
  const changed = next.length !== before.length || next.some((a, i) => a.id !== before[i]?.id)
  if (changed) {
    await update($, agents, () => next)
    const left = next.length + Object.keys(await read($, tasks)).length
    for (const a of before.filter(b => !next.some(n => n.id === b.id))) {
      $.ui.toast(`agent finished: ${clip(a.label, 40)} (${left} left)`)
    }
  }
  if (next.length > 0 || Object.keys(await read($, tasks)).length > 0) await update($, now, () => t)
}

const label = (...xs: (string | undefined)[]) => xs.find(x => x && x.trim())?.trim() ?? 'task'

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'bg-tasks',
      description: 'List running agents and background tasks (args: hide, show, clear)',
    })
    $.clock.every(POLL_MS, () => void pollAgents($).catch(() => {}))
    // Versions before 0.1.3 also pinned the count to the status line, which outlives a reload
    // and which VS Code draws as a box of its own; the count now lives in the band only.
    $.ui.status(undefined)

    return next(e)
  })

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const ran = await next(e)
    const id = ran.deny === undefined && !ran.isError ? ran.result?.backgroundTaskId : undefined
    if (id) await add($, { id, kind: 'shell', label: label(e.description, e.command.split('\n')[0]) })

    return ran
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('tool.call', { tool: 'Monitor' }, async ($, e, next) => {
    const ran = await next(e)
    const id = ran.deny === undefined && !ran.isError ? ran.result?.taskId : undefined
    if (id) await add($, { id, kind: 'monitor', label: label(e.description) })

    return ran
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('tool.call', { tool: 'Workflow' }, async ($, e, next) => {
    const ran = await next(e)
    const r = ran.deny === undefined && !ran.isError ? ran.result : undefined
    if (r?.taskId) {
      const kind = r.status === 'remote_launched' ? 'remote' : 'workflow'
      await add($, { id: r.taskId, kind, label: label(r.workflowName, e.name) })
    }

    return ran
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('tool.call', { tool: 'Agent' }, async ($, e, next) => {
    const ran = await next(e)
    const r = ran.deny === undefined && !ran.isError ? ran.result : undefined
    if (r && 'status' in r && r.status === 'remote_launched') {
      await add($, { id: r.taskId, kind: 'remote', label: label(r.description) })
    }

    return ran
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('tool.call', { tool: 'TaskStop' }, async ($, e, next) => {
    const ran = await next(e)
    const id = ran.deny === undefined && !ran.isError ? ran.result?.task_id : undefined
    if (id) await remove($, [id])

    return ran
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('prompt.submit', async ($, e, next) => {
    if (e.origin.kind === 'task-notification') {
      const map = await read($, tasks)
      const done = parseNotices(e.text).filter(n => isFinished(n, map[n.taskId]?.kind === 'monitor'))
      await remove($, done.map(n => n.taskId))
    }

    return next(e)
  }).catch(($, e, next) => next(e)) // an observer never blocks: replay what next settled to

  on('command.run', { command: 'bg-tasks' }, async ($, e) => {
    const arg = e.args.trim()
    if (arg === 'clear') {
      await update($, tasks, () => ({}))
      return { text: 'Cleared the background task list (agents are read live and stay).' }
    }
    if (arg === 'hide' || arg === 'show') {
      await update($, isHidden, () => arg === 'hide')
      return { text: `Background task band ${arg === 'hide' ? 'hidden' : 'shown'}.` }
    }
    await update($, isHidden, () => false)
    const list = await all($)
    const t = await $.clock.now()
    if (list.length === 0) return { text: 'Nothing running in the background.' }

    return {
      text: [
        `${list.length} running: ${breakdown(list)}`,
        ...list.map(x => `  ${ICON[x.kind]} ${x.label}  ${elapsed(t - x.startedAt)}  (${x.id})`),
      ].join('\n'),
    }
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const list = await all($)
    if (list.length === 0 || e.props.hasSurvey || (await read($, isHidden))) return next(e)

    const { Box, Text } = $.ui.resolve(e)
    const t = Math.max(await read($, now), ...list.map(x => x.startedAt))
    const width = Math.max(10, e.props.bodyColumns - 24)
    const rows = Math.max(0, Math.min(SHOWN, e.props.maxRows - 1))
    // The other mods' lines (and the engine's own) stay in the same band, under ours.
    const below = await next(e)

    // Laid out like usage-bar's rows: icon, a label padded to 14, the value in colour, dim details.
    return (
      <Box flexDirection="column">
        <Box>
          <Text size="caption">⟳ {'Background'.padEnd(14)} </Text>
          <Text size="caption" tabular color={GREEN}>
            {list.length} running
          </Text>
          <Text size="caption" dimColor> · {breakdown(list)}</Text>
        </Box>
        {list.slice(0, rows).map(x => (
          <Box key={x.id}>
            <Text size="caption">{`  ${ICON[x.kind]}`.padEnd(16)} </Text>
            <Text size="caption">{clip(x.label, width)}</Text>
            <Text size="caption" tabular dimColor> · {elapsed(t - x.startedAt)}</Text>
          </Box>
        ))}
        {list.length > rows && rows > 0 && (
          <Text size="caption" dimColor>
            {''.padEnd(16)} +{list.length - rows} more · /bg-tasks
          </Text>
        )}
        {below}
      </Box>
    )
  })
}
