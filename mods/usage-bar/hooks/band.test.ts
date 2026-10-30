import type { RenderElement } from 'claude-code'
import { test, expect, mock } from 'claude-code/testing'

import { GREEN, RED } from './usage'

const text = (ui: { find: (q: { type: string; text: RegExp }) => Promise<{ props: Record<string, unknown> } | undefined> }, re: RegExp) =>
  ui.find({ type: 'Text', text: re })

test('band draws both bars in their colours', async ($, on) => {
  mock.clock(on, { now: Date.parse('2026-10-20T12:00:00Z') })
  on('ui.render', ($, e) => h($.ui.resolve(e).Text, {}, 'another mod') as RenderElement)
  on('session.measure', ($, e) => ({ changed: e.changed }))
  await $.session.measure({
    context: { window: 200000 } as any,
    rateLimits: [
      { kind: 'five_hour', percentUsed: 48, resetsAt: '2026-10-20T15:00:00Z' },
      { kind: 'seven_day', percentUsed: 92, resetsAt: '2026-10-21T10:00:00Z' },
    ],
    changed: ['rateLimits'],
  })
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({
      plugin: 'usage-bar', surface, component: 'AbovePrompt',
      props: { hasSurvey: false, isWorking: false, maxRows: 10 } as any,
    })
    expect((await text(ui, /48%/))?.props.color).toBe(GREEN)
    expect((await text(ui, /92%/))?.props.color).toBe(RED)
    expect((await text(ui, /48%/))?.props.size).toBe('caption')
    expect(await text(ui, /◔ Session \(5hr\)/)).toBeDefined()
    expect(await text(ui, /Weekly \(7 day\)/)).toBeDefined()
    expect(await text(ui, /Resets in 3h/)).toBeDefined()
    expect(await text(ui, /Resets in 22h/)).toBeDefined()
    expect(await text(ui, /another mod/)).toBeDefined()
    await ui.unmount()
  }
})
