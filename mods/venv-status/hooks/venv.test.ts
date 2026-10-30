import type { RenderElement, RenderPropsOf } from 'claude-code'
import { test, expect, mock } from 'claude-code/testing'
import { describeVenv, isActiveHere, parsePyvenvVersion, venvLabel } from './venv.ts'

const P = '/Users/ani/Projects/martech_adtech_data'
const BAND = { plugin: 'venv-status', component: 'AbovePrompt' } as const
const BAND_PROPS = { hasSurvey: false, isWorking: false, maxRows: 8, bodyColumns: 80 } as RenderPropsOf['AbovePrompt']

/** The line the band above the prompt shows, on a surface. */
const bandLine = async ($: Parameters<Parameters<typeof test>[1]>[0], surface: 'terminal' | 'desktop') => {
  const band = await $.ui.mount({ ...BAND, surface, props: BAND_PROPS })
  const text = await band.find({ type: 'Text', text: /venv|Virtual Environment/ })
  await band.unmount()
  return text
}

test('labels a generic .venv by its project folder', () => {
  expect(venvLabel(`${P}/.venv`)).toBe('martech_adtech_data/.venv')
  expect(venvLabel('/Users/ani/envs/flinks_practice')).toBe('flinks_practice')
})

test('reads the version from pyvenv.cfg files', () => {
  expect(parsePyvenvVersion('home = /opt/homebrew/bin\nversion = 3.11.6\n')).toBe('3.11.6')
  expect(parsePyvenvVersion('implementation = CPython\nversion_info = 3.12.4\n')).toBe('3.12.4')
  expect(parsePyvenvVersion('nothing here')).toBe(undefined)
})

test('the project venv is active', () => {
  expect(describeVenv({ cwd: P, virtualEnv: `${P}/.venv`, projectVenv: `${P}/.venv`, activeVersion: '3.11.6' }))
    .toBe('Virtual Environment: martech_adtech_data/.venv · Python 3.11.6')
})

test('warns when another project venv is active', () => {
  expect(describeVenv({ cwd: P, virtualEnv: '/Users/ani/Projects/flinks_practice/.venv', projectVenv: `${P}/.venv` }))
    .toBe('⚠ Virtual Environment: flinks_practice/.venv · this project has its own .venv')
})

test('warns when the project venv exists but is not activated', () => {
  expect(describeVenv({ cwd: P, projectVenv: `${P}/.venv` }))
    .toBe('⚠ no venv active · martech_adtech_data/.venv exists but is not activated')
})

test('falls back to system Python', () => {
  expect(describeVenv({ cwd: P, systemVersion: '3.11.6' })).toBe('🐍 no venv · system Python 3.11.6')
})

test('refresh reads the environment and disk, then shows the line in the band', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: '/Users/ani/Projects/flinks_practice/.venv' })
  on('session.cwd', () => ({ value: P }))
  on('fs.exists', ($, e) => ({ value: e.path === `${P}/.venv/pyvenv.cfg` }))
  on('fs.read', ($, e) => {
    if (e.path === '/Users/ani/Projects/flinks_practice/.venv/pyvenv.cfg') return { value: 'version = 3.10.13\n' }
    throw new Error('missing')
  })
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Box, {}) as RenderElement)

  on('classic.CwdChanged', () => ({}))
  await $.classic.CwdChanged({ old_cwd: '/Users/ani', new_cwd: P })
  for (const surface of ['terminal', 'desktop'] as const) {
    const line = await bandLine($, surface)
    expect(line?.text).toBe('⚠ Virtual Environment: flinks_practice/.venv · Python 3.10.13 · this project has its own .venv')
    expect(line?.props).toMatchObject({ size: 'caption', color: '#FFC107' })
  }
})

test('refresh asks python3 only when nothing else is found', async ($, on) => {
  mock.env(on, {})
  on('session.cwd', () => ({ value: '/Users/ani/scratch' }))
  on('fs.exists', () => ({ value: false }))
  const ran: string[][] = []
  on('process.run', ($, e) => {
    ran.push([...e.argv])
    return { value: { exitCode: 0, stdout: '3.11.6\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
  })
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Box, {}) as RenderElement)
  on('classic.CwdChanged', () => ({}))

  await $.classic.CwdChanged({ old_cwd: '/Users/ani', new_cwd: '/Users/ani/scratch' })
  expect((await bandLine($, 'terminal'))?.text).toBe('🐍 no venv · system Python 3.11.6')
  expect(ran.length).toBe(1)
  expect(ran[0]?.[0]).toBe('python3')
})

test('/venv prints the line', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: `${P}/.venv` })
  on('session.cwd', () => ({ value: P }))
  on('fs.exists', ($, e) => ({ value: e.path === `${P}/.venv/pyvenv.cfg` }))
  on('fs.read', () => ({ value: 'version = 3.11.6\n' }))
  on('ui.log', () => ({ value: undefined }))

  const out = await $.command.run({
    command: 'venv', args: '', origin: { kind: 'sdk' }, presentation: { isFullscreen: false, columns: 80 },
  })
  expect(out.text).toBe('Virtual Environment: martech_adtech_data/.venv · Python 3.11.6')
})

test('session start clears the status line older versions pinned', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: `${P}/.venv` })
  on('session.cwd', () => ({ value: P }))
  on('fs.exists', () => ({ value: false }))
  on('fs.read', () => ({ value: 'version = 3.11.6\n' }))
  on('ui.log', () => ({ value: undefined }))
  on('command.register', () => ({ value: { command: 'venv' } }))
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  const pinned: (string | undefined)[] = []
  on('ui.status', ($, e) => { pinned.push(e.text); return { value: undefined } })

  await $.session.start({ cwd: P } as Parameters<typeof $.session.start>[0])
  expect(pinned).toEqual([undefined])
})

test('the band keeps what the plugins beneath it draw', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: `${P}/.venv` })
  on('session.cwd', () => ({ value: P }))
  on('fs.exists', ($, e) => ({ value: e.path === `${P}/.venv/pyvenv.cfg` }))
  on('fs.read', () => ({ value: 'version = 3.11.6\n' }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Text, {}, 'another mod') as RenderElement)
  on('classic.CwdChanged', () => ({}))

  await $.classic.CwdChanged({ old_cwd: '/Users/ani', new_cwd: P })
  const band = await $.ui.mount({ ...BAND, surface: 'terminal', props: BAND_PROPS })
  expect(await band.find({ type: 'Text', text: /^Virtual Environment: martech_adtech_data\/\.venv/ })).toBeDefined()
  expect(await band.find({ type: 'Text', text: 'another mod' })).toBeDefined()
  await band.unmount()
})

test('the venv counts as active here only when it belongs to the working folder', () => {
  expect(isActiveHere({ cwd: P, virtualEnv: `${P}/.venv`, projectVenv: `${P}/.venv` })).toBe(true)
  expect(isActiveHere({ cwd: `${P}/notebooks`, virtualEnv: `${P}/.venv` })).toBe(true)
  expect(isActiveHere({ cwd: P, virtualEnv: '/Users/ani/Projects/flinks_practice/.venv', projectVenv: `${P}/.venv` })).toBe(false)
  expect(isActiveHere({ cwd: '/Users/ani/scratch', virtualEnv: `${P}/.venv` })).toBe(false)
  expect(isActiveHere({ cwd: P, projectVenv: `${P}/.venv` })).toBe(false)
})

test('the label is green when the project venv is active', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: `${P}/.venv` })
  on('session.cwd', () => ({ value: P }))
  on('fs.exists', ($, e) => ({ value: e.path === `${P}/.venv/pyvenv.cfg` }))
  on('fs.read', () => ({ value: 'version = 3.11.6\n' }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Box, {}) as RenderElement)
  on('classic.CwdChanged', () => ({}))

  await $.classic.CwdChanged({ old_cwd: '/Users/ani', new_cwd: P })
  for (const surface of ['terminal', 'desktop'] as const) {
    const band = await $.ui.mount({ ...BAND, surface, props: BAND_PROPS })
    const label = await band.find({ type: 'Text', text: /^Virtual Environment$/ })
    expect(label?.props).toMatchObject({ color: '#4EBA65' })
    await band.unmount()
  }
})

test('the label is not green when the venv belongs to another folder', async ($, on) => {
  mock.env(on, { VIRTUAL_ENV: `${P}/.venv` })
  on('session.cwd', () => ({ value: '/Users/ani/scratch' }))
  on('fs.exists', () => ({ value: false }))
  on('fs.read', () => ({ value: 'version = 3.11.6\n' }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => h($.ui.resolve(e).Box, {}) as RenderElement)
  on('classic.CwdChanged', () => ({}))

  await $.classic.CwdChanged({ old_cwd: '/Users/ani', new_cwd: '/Users/ani/scratch' })
  const band = await $.ui.mount({ ...BAND, surface: 'terminal', props: BAND_PROPS })
  expect(await band.find({ type: 'Text', text: /^Virtual Environment$/ })).toBe(undefined)
  const line = await band.find({ type: 'Text', text: /^Virtual Environment: martech_adtech_data/ })
  expect(line?.props.color).toBe(undefined)
  await band.unmount()
})
