/** The line the band above the prompt shows; undefined until the first read. */
export type VenvLine = { text: string; activeHere: boolean } | undefined

declare module 'claude-code' {
  interface PluginState {
    'venv-status': { line: VenvLine }
  }
}
