export type BgKind = 'agent' | 'shell' | 'monitor' | 'workflow' | 'remote'
export type BgTask = { id: string; kind: BgKind; label: string; startedAt: number }

declare module 'claude-code' {
  interface PluginState {
    'bg-tasks': {
      /** Shell, monitor, workflow and remote tasks, keyed by task id. */
      tasks: Record<string, BgTask>
      /** Agents `$.agent.list()` last reported as live. */
      agents: BgTask[]
      /** Last clock tick, so elapsed times redraw. */
      now: number
      isHidden: boolean
    }
  }
}
