import { expect, test } from 'claude-code/testing'

import { buildPrompt } from './register'

// The test environment has timers; the types name no DOM or Node globals.
declare const setTimeout: (callback: () => void, ms: number) => unknown

const PANE = {
  plugin: 'side-chat',
  component: 'Pane',
  requestId: 'side-chat',
  props: {
    title: 'Side chat',
    isFocused: true,
    bodyColumns: 60,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 30 },
    view: {},
  },
} as const

const USAGE = {
  input_tokens: 10,
  output_tokens: 5,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
}

const QUESTION = {
  tool: 'AskUserQuestion' as const,
  questions: [
    {
      question: 'Which split?',
      header: 'Split',
      multiSelect: false,
      options: [
        { label: 'By time', description: 'rolling folds' },
        { label: 'Random', description: 'stratified' },
      ],
    },
  ],
}

// The pane's handler runs the fork without awaiting it: wait for the redraw.
const settle = async (check: () => Promise<boolean>) => {
  for (let i = 0; i < 100; i += 1) {
    if (await check()) {
      return
    }
    await new Promise<void>(resolve => setTimeout(resolve, 5))
  }
  throw new Error('never settled')
}

test('the prompt carries the pending question, earlier turns and the new one', () => {
  const prompt = buildPrompt(
    'why time?',
    [
      { role: 'you', text: 'what is a fold?' },
      { role: 'error', text: 'api error' },
      { role: 'side', text: 'a slice of training data' },
    ],
    [
      {
        question: 'Which split?',
        isMultiSelect: false,
        options: [
          { label: 'By time', description: 'rolling folds' },
          { label: 'Random', description: '' },
        ],
      },
      {
        question: 'Which metrics?',
        isMultiSelect: true,
        options: [{ label: 'MAE', description: 'in degrees' }],
      },
    ],
  )

  expect(prompt).toContain('Which split? (pick one)\n- By time: rolling folds\n- Random\n\nWhich metrics? (pick any)\n- MAE: in degrees')
  expect(prompt).toContain('Person: what is a fold?\nSide chat: a slice of training data')
  expect(prompt).not.toContain('api error')
  expect(prompt.endsWith('Person: why time?')).toBe(true)
})

test('/side opens the pane', async ($, on) => {
  const opened: string[] = []
  on('command.register', () => ({ value: { command: 'side' } }))
  on('ui.open', (_, e) => {
    opened.push(e.id)
    return { value: { isPlaced: true } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))

  await $.session.start({ cwd: '/', surface: 'desktop', isInteractive: true })
  await $.command.run({ command: 'side', args: '' } as Parameters<typeof $.command.run>[0])

  expect(opened).toEqual(['side-chat'])
})

test('a waiting question opens the pane and is cleared once answered', async ($, on) => {
  const opened: string[] = []
  let shownWhileWaiting = ''
  on('ui.open', (_, e) => {
    opened.push(e.id)
    return { value: { isPlaced: true } }
  })
  on('tool.call', async () => {
    const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
    shownWhileWaiting = (await ui.find({ type: 'Text', text: /Waiting on/ }))?.text ?? ''
    await ui.unmount()
    return { result: {} }
  })

  await $.tool.call(QUESTION)

  expect(opened).toEqual(['side-chat'])
  expect(shownWhileWaiting).toBe('Waiting on: Which split?')
  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  expect(await ui.find({ type: 'Text', text: /Waiting on/ })).toBeUndefined()
  await ui.unmount()
})

test('a side question asked while one is waiting carries every question and description', async ($, on) => {
  let prompt = ''
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('model.fork', (_, e) => {
    prompt = e.prompt
    return answered('Time keeps the future out.')
  })
  on('tool.call', async () => {
    const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
    await ui.input({ key: 'question', text: 'which one?' })
    await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)
    await ui.unmount()
    return { result: {} }
  })

  await $.tool.call({
    ...QUESTION,
    questions: [
      ...QUESTION.questions,
      {
        question: 'Which metrics?',
        header: 'Metrics',
        multiSelect: true,
        options: [
          { label: 'MAE', description: 'in degrees' },
          { label: 'RMSE', description: 'punishes big misses' },
        ],
      },
    ],
  })

  expect(prompt).toContain('Which split? (pick one)\n- By time: rolling folds\n- Random: stratified')
  expect(prompt).toContain('Which metrics? (pick any)\n- MAE: in degrees\n- RMSE: punishes big misses')
})

test('a quick question button asks its question, and hides while busy', async ($, on) => {
  const prompts: string[] = []
  let release = () => {}
  on('model.fork', async (_, e) => {
    prompts.push(e.prompt)
    await new Promise<void>(resolve => {
      release = resolve
    })
    return answered('Most teams use an expanding window.')
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.press({ key: 'quick-1' })
  await settle(async () => prompts.length === 1)
  expect(await ui.find({ key: 'quick-2' })).toBeUndefined()

  release()
  await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)
  expect(prompts[0]?.endsWith("Person: What's the standard industry pattern?")).toBe(true)
  expect((await ui.find({ key: 'quick-3' }))?.props.label).toBe('Explain all options')
  await ui.unmount()
})

test('a question gets a side answer, or an error line', async ($, on) => {
  const prompts: string[] = []
  on('model.fork', (_, e) => {
    prompts.push(e.prompt)
    return prompts.length === 1
      ? { value: { isAnswered: true, text: 'Because time leaks.', usage: USAGE } }
      : { value: { isAnswered: false, reason: 'nothing-to-fork', usage: USAGE } }
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  await ui.input({ key: 'question', text: 'why time?' })
  await settle(async () => (await ui.find({ key: 'question' })) !== undefined && prompts.length === 1)
  await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)
  expect((await ui.find({ type: 'Markdown' }))?.text).toContain('Because time leaks.')

  await ui.input({ key: 'question', text: 'and random?' })
  await settle(async () => (await ui.find({ type: 'Text', text: /Nothing to ask about/ })) !== undefined)
  expect(prompts[1]).toContain('Side chat: Because time leaks.')
  await ui.unmount()
})

test('Discard drops the late reply and frees the input', async ($, on) => {
  let release = () => {}
  on('model.fork', async () => {
    await new Promise<void>(resolve => {
      release = resolve
    })
    return { value: { isAnswered: true, text: 'late reply', usage: USAGE } }
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'slow one' })
  await settle(async () => (await ui.find({ key: 'discard' })) !== undefined)
  await ui.press({ key: 'discard' })
  await settle(async () => (await ui.find({ key: 'question' })) !== undefined)

  release()
  await new Promise<void>(resolve => setTimeout(resolve, 50))
  expect(await ui.find({ type: 'Markdown' })).toBeUndefined()
  await ui.unmount()
})

test('Clear empties the log', async ($, on) => {
  on('model.fork', () => ({ value: { isAnswered: true, text: 'an answer', usage: USAGE } }))

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'hello' })
  await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)
  await ui.press({ key: 'clear' })
  await settle(async () => (await ui.find({ type: 'Markdown' })) === undefined)
  expect((await ui.find({ type: 'Text', text: /without interrupting the conversation/ }))?.text).toBeDefined()
  await ui.unmount()
})

const answered = (text: string) => ({ value: { isAnswered: true as const, text, usage: USAGE } })

test('a SEARCH reply runs a web search, then the next fork answers with its results', async ($, on) => {
  const prompts: string[] = []
  const queries: string[] = []
  on('model.fork', (_, e) => {
    prompts.push(e.prompt)
    return prompts.length === 1
      ? answered('SEARCH: purged k-fold time series')
      : answered('Purged k-fold drops overlapping rows. https://example.org/purged')
  })
  on('tool.call', (_, e) => {
    if (e.tool === 'WebSearch') {
      queries.push(e.query)
    }
    return { result: { query: 'q', results: [], durationSeconds: 1 }, text: 'Purged CV - https://example.org/purged' }
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'is purged k-fold right here?' })
  await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)

  expect(queries).toEqual(['purged k-fold time series'])
  expect(prompts[0]).toContain('SEARCH: <query>')
  expect(prompts[1]).toContain('Web results for "purged k-fold time series":\nPurged CV - https://example.org/purged')
  expect((await ui.find({ type: 'Text', text: /searching:/ }))?.text).toBe('searching: purged k-fold time series')
  expect((await ui.find({ type: 'Markdown' }))?.text).toContain('Purged k-fold drops overlapping rows.')
  await ui.unmount()
})

test('searches stop at two, and the last fork is told to answer now', async ($, on) => {
  const prompts: string[] = []
  let searches = 0
  on('model.fork', (_, e) => {
    prompts.push(e.prompt)
    return answered(`SEARCH: query ${prompts.length}`)
  })
  on('tool.call', () => {
    searches += 1
    return { result: { query: 'q', results: [], durationSeconds: 1 }, text: 'hits' }
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'keep searching' })
  await settle(async () => (await ui.find({ type: 'Text', text: /Stopped after 2 searches/ })) !== undefined)

  expect(searches).toBe(2)
  expect(prompts).toHaveLength(3)
  expect(prompts[2]).toContain('no more searches')
  expect(await ui.find({ key: 'question' })).toBeDefined()
  await ui.unmount()
})

test('a refused search shows an error line and the side chat still answers', async ($, on) => {
  const prompts: string[] = []
  on('model.fork', (_, e) => {
    prompts.push(e.prompt)
    return prompts.length === 1 ? answered('SEARCH: anything') : answered('Answered without the web.')
  })
  on('tool.call', () => ({ deny: 'no web here' }))

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'look it up' })
  await settle(async () => (await ui.find({ type: 'Markdown' })) !== undefined)

  expect((await ui.find({ type: 'Text', text: /refused/ }))?.text).toContain('no web here')
  expect(prompts[1]).toContain('The search was refused: no web here')
  expect((await ui.find({ type: 'Markdown' }))?.text).toContain('Answered without the web.')
  await ui.unmount()
})

test('Discard during a search starts no further fork', async ($, on) => {
  let forks = 0
  let release = () => {}
  on('model.fork', () => {
    forks += 1
    return answered('SEARCH: slow')
  })
  on('tool.call', async () => {
    await new Promise<void>(resolve => {
      release = resolve
    })
    return { result: { query: 'q', results: [], durationSeconds: 1 }, text: 'hits' }
  })

  const ui = await $.ui.mount({ ...PANE, surface: 'desktop' })
  await ui.input({ key: 'question', text: 'slow search' })
  await settle(async () => (await ui.find({ type: 'Text', text: /searching:/ })) !== undefined)
  await ui.press({ key: 'discard' })
  release()
  await new Promise<void>(resolve => setTimeout(resolve, 50))

  expect(forks).toBe(1)
  expect(await ui.find({ key: 'question' })).toBeDefined()
  await ui.unmount()
})
