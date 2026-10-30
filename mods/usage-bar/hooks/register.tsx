import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Limit } from '../types'
import { LABELS, bar, colorFor, resetsIn, shown } from './usage'

const limits = atom({ plugin: 'usage-bar', key: 'limits' } as const, [] as Limit[])

const keep = (rateLimits: Limit[]) =>
  rateLimits.map(({ kind, percentUsed, resetsAt }) => ({ kind, percentUsed, resetsAt }))

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const result = await next(e)
    const usage = await $.session.usage()
    await update($, limits, () => keep(usage.rateLimits))
    return result
  })

  on('session.measure', async ($, e, next) => {
    await update($, limits, () => keep(e.rateLimits))
    return next(e)
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const rows = shown(await read($, limits))
    if (e.props.hasSurvey || rows.length === 0) {
      return next(e)
    }

    const { Box, Text } = $.ui.resolve(e)
    const now = await $.clock.now()
    // The other mods' lines (and the engine's own) stay in the same band, under ours.
    const below = await next(e)

    return (
      <Box flexDirection="column">
        {rows.map(l => (
          <Box key={l.kind}>
            <Text size="caption">◔ {(LABELS[l.kind] ?? l.kind).padEnd(14)} </Text>
            <Text size="caption" tabular color={colorFor(l.percentUsed)}>
              {bar(l.percentUsed)} {Math.round(l.percentUsed)}%
            </Text>
            {l.resetsAt && <Text size="caption" dimColor> · {resetsIn(l.resetsAt, now)}</Text>}
          </Box>
        ))}
        {below}
      </Box>
    )
  })
}
