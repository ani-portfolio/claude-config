export type Turn = { role: 'you' | 'side' | 'search' | 'error'; text: string }

export type Option = { label: string; description: string }

// One question of the AskUserQuestion waiting on the person, as the side chat
// is told it.
export type Pending = { question: string; isMultiSelect: boolean; options: Option[] }

declare module 'claude-code' {
  interface PluginState {
    'side-chat': {
      turns: Shaped<Turn[]>
      // Every question of the waiting call; null when none is waiting.
      pending: Shaped<Pending[] | null>
      // The id of the side question in flight; null when none is.
      busy: number | null
    }
  }
}
