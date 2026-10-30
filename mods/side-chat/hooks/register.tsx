import { atom, read, update } from 'claude-code'
import type { EngineInterface, ModelForkResult, Register } from 'claude-code'

import type { Pending, Turn } from '../types'

const PANE = 'side-chat'
const TITLE = 'Side chat'
const MAX_TURNS = 50
// Earlier side turns sent with each question, so follow-ups have context.
const CONTEXT_TURNS = 10
const MAX_SEARCHES = 2
const MAX_RESULT_CHARS = 4000
const SEARCH_LINE = /^SEARCH:\s*(.+)$/

// One-press side questions the person asks often, sent as written.
const QUICK_QUESTIONS = [
  "What's the standard industry pattern?",
  'Explain the recommended option',
  'Explain all options',
]

// Change a tag when that value's fields change: it declines what an earlier
// version stored, which outlives a reload.
const turns = atom({ plugin: 'side-chat', key: 'turns' } as const, [], {
  shape: 'turns-1',
})
const pending = atom({ plugin: 'side-chat', key: 'pending' } as const, null, {
  shape: 'pending-2',
})
const busy = atom({ plugin: 'side-chat', key: 'busy' } as const, null)

const PREAMBLE = [
  'You are a side chat beside the conversation above. The person is asking',
  'you about it without adding anything to the main thread. Answer briefly',
  'and plainly. You cannot use tools, and nothing you say reaches the main',
  'thread.',
].join(' ')

const MAY_SEARCH = [
  'If you need facts from the web to answer well, reply with exactly one line,',
  '`SEARCH: <query>`, and nothing else; the side chat runs it and asks you',
  'again with the results. Otherwise answer now. When you use web results, end',
  'with the URLs you relied on.',
].join(' ')

const NO_MORE_SEARCH =
  'Answer now from what you have; no more searches. End with the URLs you relied on.'

export type Search = { query: string; text: string; isFailed?: true }

export const buildPrompt = (
  question: string,
  earlier: readonly Turn[],
  waiting: readonly Pending[] | null,
  searches: readonly Search[] = [],
): string => {
  const parts = [PREAMBLE, searches.length < MAX_SEARCHES ? MAY_SEARCH : NO_MORE_SEARCH]

  if (waiting !== null && waiting.length > 0) {
    const asked = waiting.map(item => {
      const pick = item.isMultiSelect ? 'pick any' : 'pick one'
      const options = item.options
        .map(option => `- ${option.label}${option.description === '' ? '' : `: ${option.description}`}`)
        .join('\n')
      return `${item.question} (${pick})\n${options}`
    })
    parts.push(
      `The assistant is waiting on the person to answer:\n\n${asked.join('\n\n')}`,
    )
  }

  const said = earlier
    .filter(turn => turn.role !== 'error')
    .slice(-CONTEXT_TURNS)
    .map(turn =>
      turn.role === 'search'
        ? `(searched the web: ${turn.text})`
        : `${turn.role === 'you' ? 'Person' : 'Side chat'}: ${turn.text}`,
    )
  if (said.length > 0) {
    parts.push(`The side chat so far:\n${said.join('\n')}`)
  }

  parts.push(`Person: ${question}`)

  for (const search of searches) {
    parts.push(`Web results for "${search.query}":\n${search.text}`)
  }

  return parts.join('\n\n')
}

const describeFailure = (result: ModelForkResult): string => {
  if (result.isAnswered) {
    return ''
  }
  if (result.reason === 'nothing-to-fork') {
    return 'Nothing to ask about yet: the main thread has not sent a request.'
  }
  if (result.reason === 'api-error') {
    return `The API answered with an error (${result.error}, status ${result.status ?? 'none'}).`
  }

  return `No reply (${result.reason}).`
}

const addTurn = ($: EngineInterface, turn: Turn) =>
  update($, turns, list => [...list, turn].slice(-MAX_TURNS))

// Runs one web search; a failure comes back as text the model reads.
const search = async ($: EngineInterface, query: string): Promise<Search> => {
  try {
    const ran = await $.tool.call({ tool: 'WebSearch', query, mode: 'standard' })
    if (ran.deny !== undefined) {
      return { query, text: `The search was refused: ${ran.deny}`, isFailed: true }
    }
    return { query, text: (ran.text ?? '').slice(0, MAX_RESULT_CHARS) }
  } catch (error) {
    return { query, text: `The search failed: ${String(error)}`, isFailed: true }
  }
}

const ask = async ($: EngineInterface, text: string) => {
  const question = text.trim()
  if (question === '' || (await read($, busy)) !== null) {
    return
  }

  const earlier = await read($, turns)
  const waiting = await read($, pending)
  const id = Math.random()
  await addTurn($, { role: 'you', text: question })
  await update($, busy, () => id)

  // Discard set busy to null (or a newer question took its place): stop.
  const isDiscarded = async () => (await read($, busy)) !== id
  const searches: Search[] = []

  for (;;) {
    const result = await $.model.fork({
      prompt: buildPrompt(question, earlier, waiting, searches),
    })
    if (await isDiscarded()) {
      return
    }

    const wanted = result.isAnswered ? SEARCH_LINE.exec(result.text.trim()) : null
    if (wanted !== null && searches.length < MAX_SEARCHES) {
      const query = (wanted[1] ?? '').trim()
      await addTurn($, { role: 'search', text: query })
      const found = await search($, query)
      if (await isDiscarded()) {
        return
      }
      if (found.isFailed) {
        await addTurn($, { role: 'error', text: found.text })
      }
      searches.push(found)
      continue
    }

    await addTurn(
      $,
      wanted !== null
        ? { role: 'error', text: `Stopped after ${MAX_SEARCHES} searches without an answer.` }
        : result.isAnswered
          ? { role: 'side', text: result.text }
          : { role: 'error', text: describeFailure(result) },
    )
    await update($, busy, () => null)
    return
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'side',
      description: 'Open the side chat about this thread',
    })

    return next(e)
  })

  on('command.run', { command: 'side' }, async $ => {
    await $.ui.open({ id: PANE, title: TITLE })

    return { text: 'Side chat opened.' }
  })

  on('tool.call', { tool: 'AskUserQuestion' }, async ($, e, next) => {
    if (e.questions.length > 0) {
      const asked = e.questions.map(item => ({
        question: item.question,
        isMultiSelect: item.multiSelect === true,
        options: item.options.map(option => ({
          label: option.label,
          description: option.description ?? '',
        })),
      }))
      await update($, pending, () => asked)
      void $.ui.open({ id: PANE, title: TITLE })
    }

    try {
      return await next(e)
    } finally {
      // The pane stays open after the answer; only the question is cleared.
      await update($, pending, () => null)
    }
    // A failure here must never block the question itself.
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const elements = $.ui.resolve(e)
    const { Box, Button, Markdown, Text } = elements
    // The mobile app draws no field yet.
    const Input = 'Input' in elements ? elements.Input : undefined
    const list = await read($, turns)
    const waiting = await read($, pending)
    const isBusy = (await read($, busy)) !== null
    const room = Math.max(1, (e.viewport?.rows ?? 24) - 10)

    // Styled after the desktop app's own side question pane: a header with
    // Clear, an italic hint, the log, and the input pinned at the bottom.
    return (
      <Box flexDirection="column" height="100%" gap={1}>
        <Box flexDirection="row" justifyContent="space-between" alignItems="center">
          <Text>
            <Text color="claude">✳ </Text>
            <Text weight="semibold">{TITLE}</Text>
          </Text>
          <Button key="clear" label="Clear" plain dimColor onPress={() => update($, turns, () => [])} />
        </Box>
        {list.length === 0 && (
          <Text dimColor italic>
            Ask a quick side question below without interrupting the conversation.
          </Text>
        )}
        {waiting !== null && waiting[0] !== undefined && (
          <Text dimColor italic>
            Waiting on: {waiting[0].question}
            {waiting.length > 1 ? ` (+${waiting.length - 1} more)` : ''}
          </Text>
        )}
        <Box flexDirection="column" flexGrow={1} overflow="hidden" gap={1}>
          {list.slice(-room).map(turn =>
            turn.role === 'error' ? (
              <Text color="error" dimColor>
                {turn.text}
              </Text>
            ) : turn.role === 'search' ? (
              <Text dimColor italic>
                searching: {turn.text}
              </Text>
            ) : turn.role === 'you' ? (
              <Box paddingX={1} backgroundColor="subtle">
                <Text weight="medium">{turn.text}</Text>
              </Box>
            ) : (
              <Markdown text={turn.text} />
            ),
          )}
        </Box>
        <Box flexDirection="column">
          {isBusy ? (
            <Box flexDirection="row" justifyContent="space-between" alignItems="center">
              <Text dimColor italic>
                thinking…
              </Text>
              <Button
                key="discard"
                label="Discard"
                variant="secondary"
                onPress={() => update($, busy, () => null)}
              />
            </Box>
          ) : (
            <Box flexDirection="column" gap={1}>
              <Box flexDirection="row" flexWrap="wrap" gap={1}>
                {QUICK_QUESTIONS.map((text, index) => (
                  <Button
                    key={`quick-${index + 1}`}
                    label={text}
                    hotkey={String(index + 1)}
                    variant="secondary"
                    onPress={() => void ask($, text)}
                  />
                ))}
              </Box>
              {Input === undefined ? (
                <Text dimColor>Type a question from the terminal or desktop app.</Text>
              ) : (
                <Input
                  key="question"
                  placeholder="Ask a side question"
                  submitLabel="Send"
                  value=""
                  rows={3}
                  autoFocus
                  onSubmit={(value: string) => void ask($, value)}
                />
              )}
            </Box>
          )}
        </Box>
      </Box>
    )
  })
}
