import { test, expect } from 'claude-code/testing'

import { bar, colorFor, GREEN, RED, resetsIn, shown, WIDTH, YELLOW } from './usage'

test('colour follows the thresholds', async () => {
  expect(colorFor(0)).toBe(GREEN)
  expect(colorFor(48)).toBe(GREEN)
  expect(colorFor(49.9)).toBe(GREEN)
  expect(colorFor(50)).toBe(YELLOW)
  expect(colorFor(67)).toBe(YELLOW)
  expect(colorFor(89.9)).toBe(YELLOW)
  expect(colorFor(90)).toBe(RED)
  expect(colorFor(100)).toBe(RED)
})

test('bar fills in proportion and clamps', async () => {
  expect(bar(0)).toBe('░'.repeat(WIDTH))
  expect(bar(50)).toBe('█'.repeat(10) + '░'.repeat(10))
  expect(bar(130)).toBe('█'.repeat(WIDTH))
})

test('reset time reads like the usage screen', async () => {
  const now = Date.parse('2026-10-20T12:00:00Z')
  expect(resetsIn('2026-10-20T15:00:00Z', now)).toBe('Resets in 3h')
  expect(resetsIn('2026-10-21T10:00:00Z', now)).toBe('Resets in 22h')
  expect(resetsIn('2026-10-20T12:30:00Z', now)).toBe('Resets in 30m')
  expect(resetsIn(undefined, now)).toBe('')
})

test('only session and weekly windows show, session first', async () => {
  const rows = shown([
    { kind: 'seven_day', percentUsed: 67 },
    { kind: 'spend_limit', percentUsed: 5 },
    { kind: 'five_hour', percentUsed: 48 },
  ])
  expect(rows.map(r => r.kind)).toEqual(['five_hour', 'seven_day'])
})
