import type { CommandRunInput, On, RenderElement, RenderPropsOf } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

import { isFinished, parseNotices } from './parse'

const BAND = { plugin: 'bg-tasks', component: 'AbovePrompt' } as const
// The band reads only these; the site's scroll and view are the engine's.
const BAND_PROPS = { hasSurvey: false, isWorking: false, maxRows: 8, bodyColumns: 80 } as RenderPropsOf['AbovePrompt']
// The engine stamps origin and presentation on a typed command.
const BG_TASKS = { command: 'bg-tasks', args: '' } as CommandRunInput

/** The engine beneath the plugin: records the status line, swallows toasts, takes the command. */
const world = (on: On) => {
  const status: (string | undefined)[] = []
  on('ui.status', ($, e) => {
    status.push(e.text)
    return { value: undefined }
  })
  on('ui.toast', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'bg-tasks' } }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Box, {}) as RenderElement)
  return { status, last: () => status.at(-1) }
}

/** The band's Text matching `text` on the terminal, or undefined. */
const bandText = async ($: Parameters<Parameters<typeof test>[1]>[0], text: RegExp) => {
  const band = await $.ui.mount({ ...BAND, surface: 'terminal', props: BAND_PROPS })
  const found = await band.find({ text })
  await band.unmount()
  return found
}

const notice = (id: string, status?: string) =>
  `<task-notification>\n<task-id>${id}</task-id>\n${status ? `<status>${status}</status>\n` : ''}<summary>done</summary>\n</task-notification>`

test('parses one or several notices, and only terminal ones finish a monitor', async () => {
  expect(parseNotices(notice('b1', 'completed'))).toEqual([{ taskId: 'b1', status: 'completed' }])
  expect(parseNotices(notice('a', 'failed') + notice('b', 'killed')).map(n => n.taskId)).toEqual(['a', 'b'])
  expect(parseNotices('plain text')).toEqual([])
  expect(isFinished({ taskId: 'm', status: 'completed' }, true)).toBe(true)
  expect(isFinished({ taskId: 'm' }, true)).toBe(false)
  expect(isFinished({ taskId: 's' }, false)).toBe(true)
  expect(isFinished({ taskId: 's', status: 'running' }, false)).toBe(false)
})

test('a backgrounded Bash call is counted until TaskStop ends it', async ($, on) => {
  mock.clock(on, { now: 1000 })
  const ui = world(on)
  on('tool.call', { tool: 'Bash' }, () => ({
    result: { stdout: '', stderr: '', interrupted: false, backgroundTaskId: 'sh1' },
  }))
  on('tool.call', { tool: 'TaskStop' }, () => ({
    result: { message: 'stopped', task_id: 'sh1', task_type: 'local_bash' },
  }))

  await $.tool.call({ tool: 'Bash', command: 'sleep 20', description: 'nap', run_in_background: true })
  expect(await bandText($, /1 running/)).toBeDefined()
  const listed = await $.command.run(BG_TASKS)
  expect(listed).toMatchObject({ text: expect.stringContaining('$ nap') })

  await $.tool.call({ tool: 'TaskStop', task_id: 'sh1' })
  expect(await bandText($, /running/)).toBeUndefined()
  expect(await $.command.run(BG_TASKS)).toMatchObject({ text: 'Nothing running in the background.' })
})

test('a foreground Bash call is not counted', async ($, on) => {
  mock.clock(on)
  const ui = world(on)
  on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: 'hi', stderr: '', interrupted: false } }))
  await $.tool.call({ tool: 'Bash', command: 'echo hi' })
  expect(await bandText($, /running/)).toBeUndefined()
})

test('live agents come from agent.list on each poll, and the band shows them', async ($, on) => {
  const clock = mock.clock(on, { now: 0 })
  const ui = world(on)
  let live = [
    { id: 'ag1', description: 'experiment A', type: 'ml-experimenter', status: 'running' as const },
    { id: 'ag2', description: 'old one', type: 'Explore', status: 'completed' as const },
  ]
  on('agent.list', () => ({ value: live }))
  on('tool.call', { tool: 'Bash' }, () => ({
    result: { stdout: '', stderr: '', interrupted: false, backgroundTaskId: 'sh1' },
  }))

  await $.session.start({ cwd: '/tmp', surface: 'terminal', isInteractive: true })
  await clock.advance(2000)
  expect(await bandText($, /1 running/)).toBeDefined()
  await $.tool.call({ tool: 'Bash', command: 'sleep 20', run_in_background: true })
  expect(await bandText($, /2 running/)).toBeDefined()

  for (const surface of ['terminal', 'vscode'] as const) {
    const band = await $.ui.mount({ ...BAND, surface, props: BAND_PROPS })
    expect(await band.find({ text: /2 running/ })).toBeDefined()
    expect(await band.find({ text: /1 agent · 1 shell/ })).toBeDefined()
    expect(await band.find({ text: /experiment A/ })).toBeDefined()
    await band.unmount()
  }

  live = []
  await clock.advance(2000)
  expect(await bandText($, /1 running/)).toBeDefined()
  expect(ui.status.filter(s => s !== undefined)).toEqual([])
})

test('/bg-tasks hide takes the band away and /bg-tasks brings it back', async ($, on) => {
  mock.clock(on)
  world(on)
  on('tool.call', { tool: 'Bash' }, () => ({
    result: { stdout: '', stderr: '', interrupted: false, backgroundTaskId: 'sh1' },
  }))
  await $.tool.call({ tool: 'Bash', command: 'sleep 20', run_in_background: true })

  await $.command.run({ ...BG_TASKS, args: 'hide' })
  expect(await bandText($, /running/)).toBeUndefined()

  await $.command.run(BG_TASKS)
  expect(await bandText($, /1 running/)).toBeDefined()
})

test('the band row is laid out like usage-bar, and session start takes down the old status line', async ($, on) => {
  mock.clock(on)
  const ui = world(on)
  on('agent.list', () => ({ value: [] }))
  on('tool.call', { tool: 'Bash' }, () => ({
    result: { stdout: '', stderr: '', interrupted: false, backgroundTaskId: 'sh1' },
  }))

  await $.session.start({ cwd: '/tmp', surface: 'terminal', isInteractive: true })
  expect(ui.status).toEqual([undefined])
  await $.tool.call({ tool: 'Bash', command: 'sleep 20', run_in_background: true })

  for (const surface of ['terminal', 'vscode'] as const) {
    const band = await $.ui.mount({ ...BAND, surface, props: BAND_PROPS })
    expect(await band.find({ type: 'Text', text: '⟳ Background     ' })).toBeDefined()
    expect((await band.find({ type: 'Text', text: /^1 running$/ }))?.props).toMatchObject({ color: '#4EBA65' })
    expect(await band.find({ type: 'Button' })).toBeUndefined()
    await band.unmount()
  }
})
