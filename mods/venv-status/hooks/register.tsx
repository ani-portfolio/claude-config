import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'
import type { VenvLine } from '../types'
import { describeVenv, isActiveHere, parsePyvenvVersion, type VenvFacts } from './venv.ts'

const shown = atom({ plugin: 'venv-status', key: 'line' } as const, undefined as VenvLine)

// Raw colours, not the `warning` or `success` theme keys, which the VS Code panel draws as plain text.
const WARNING = '#FFC107'
const ACTIVE = '#4EBA65' // the same green as usage-bar

const LABEL = 'Virtual Environment'

const CANDIDATES = ['.venv', 'venv', 'env', '.env']

const readVersion = async ($: EngineInterface, venv: string): Promise<string | undefined> => {
  try {
    return parsePyvenvVersion(await $.fs.read(`${venv}/pyvenv.cfg`))
  } catch {
    return undefined
  }
}

const findProjectVenv = async ($: EngineInterface, cwd: string): Promise<string | undefined> => {
  for (const name of CANDIDATES) {
    const dir = `${cwd}/${name}`
    // pyvenv.cfg is what marks a folder as a virtual environment (a .env file is not one).
    if (await $.fs.exists(`${dir}/pyvenv.cfg`)) return dir
  }
  return undefined
}

const systemPython = async ($: EngineInterface): Promise<string | undefined> => {
  try {
    const r = await $.process.run(['python3', '-c', 'import platform; print(platform.python_version())'], {
      timeoutMs: 5000,
    })
    return r.exitCode === 0 ? r.stdout.trim() || undefined : undefined
  } catch {
    return undefined
  }
}

/** Gathers the facts and shows the line in the band above the prompt. Exported so tests can call it. */
export const refresh = async ($: EngineInterface): Promise<string> => {
  const cwd = await $.session.cwd()
  const virtualEnv = (await $.env.get('VIRTUAL_ENV')) || undefined
  const condaEnv = (await $.env.get('CONDA_DEFAULT_ENV')) || undefined
  const projectVenv = await findProjectVenv($, cwd)

  const facts: VenvFacts = { cwd, virtualEnv, condaEnv, projectVenv }
  if (virtualEnv) facts.activeVersion = await readVersion($, virtualEnv)
  else if (!condaEnv && !projectVenv) facts.systemVersion = await systemPython($)

  const line = describeVenv(facts)
  const activeHere = isActiveHere(facts)
  const prev = await read($, shown)
  if (prev?.text !== line || prev.activeHere !== activeHere) await update($, shown, () => ({ text: line, activeHere }))
  return line
}

const safeRefresh = async ($: EngineInterface): Promise<string | undefined> => {
  try {
    return await refresh($)
  } catch (err) {
    $.ui.log(`venv-status: could not read the environment (${String(err)})`, { to: 'debug' })
    return undefined
  }
}

/** True when some client other than the terminal draws this session (VS Code, desktop, mobile). */
const hasAppSurface = async ($: EngineInterface) =>
  (await $.session.surfaces()).some(s => s !== 'terminal')

export const register: Register = on => {
  // The last warning shown beneath an answer, so the same one is not repeated every turn.
  let lastNote: string | undefined

  // When the session is ready (and again after each reload of the mod).
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    // Versions before 0.1.2 pinned the line to the status line, which outlives a reload;
    // the line now lives in the band, so take any old pin down.
    $.ui.status(undefined)
    await $.command.register({
      name: 'venv',
      description: 'Show the Python virtual environment this session is using',
    })
    await safeRefresh($)
    return started
  })

  // /venv prints the line in the conversation. This works in VS Code, which has no status line slot.
  on('command.run', { command: 'venv' }, async $ => {
    const line = await safeRefresh($)
    return { text: line ?? 'venv-status could not read the environment.' }
  })

  // When the working directory changes (/cd, a worktree move).
  on('classic.CwdChanged', async ($, e, next) => {
    const result = await next(e)
    await safeRefresh($)
    return result
  }).catch(($, e, next) => next(e))

  // After each turn: refresh the status line. In VS Code and the desktop app, which do not draw
  // a plugin's status line, show a warning beneath the answer instead, once per new warning.
  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (!(await hasAppSurface($))) {
      void safeRefresh($)
      return result
    }
    const line = await safeRefresh($)
    if (line?.startsWith('⚠') && line !== lastNote) {
      lastNote = line
      return { ...result, text: line }
    }
    if (!line?.startsWith('⚠')) lastNote = undefined
    return result
  })

  // One line in the band above the prompt, in the same frame as the other mods' lines:
  // ours first, then whatever the plugins beneath (and the engine) draw there.
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const state = await read($, shown)
    // Versions before 0.1.3 kept a plain string here, which can outlive a reload.
    if (typeof state !== 'object' || e.props.hasSurvey) return next(e)
    const { text: line, activeHere } = state

    const { Box, Text } = $.ui.resolve(e)
    const below = await next(e)

    return (
      <Box flexDirection="column">
        {activeHere && line.startsWith(LABEL) ? (
          <Text size="caption">
            <Text color={ACTIVE}>{LABEL}</Text>
            {line.slice(LABEL.length)}
          </Text>
        ) : (
          <Text size="caption" color={line.startsWith('⚠') ? WARNING : undefined}>
            {line}
          </Text>
        )}
        {below}
      </Box>
    )
  })
}
