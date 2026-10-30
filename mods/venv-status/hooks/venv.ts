// Pure logic: turns what was found on disk and in the environment into one status line.

export type VenvFacts = {
  cwd: string
  virtualEnv?: string // $VIRTUAL_ENV, set by `source .venv/bin/activate` or `uv run`
  condaEnv?: string // $CONDA_DEFAULT_ENV
  activeVersion?: string // Python version of the active venv, from its pyvenv.cfg
  projectVenv?: string // a .venv or venv folder inside the working directory
  systemVersion?: string // what `python3` reports when no venv is active
}

const VENV_DIR_NAMES = ['.venv', 'venv', 'env', '.env']

const trimSlash = (p: string) => (p.length > 1 ? p.replace(/\/+$/, '') : p)
const base = (p: string) => trimSlash(p).split('/').pop() ?? p
const parent = (p: string) => trimSlash(p).split('/').slice(0, -1).join('/') || '/'

/** "martech_adtech_data/.venv" for a generic folder name, else just the folder name. */
export const venvLabel = (path: string): string => {
  const name = base(path)
  return VENV_DIR_NAMES.includes(name) ? `${base(parent(path))}/${name}` : name
}

export const isInside = (child: string, dir: string): boolean => {
  const c = trimSlash(child)
  const d = trimSlash(dir)
  return c === d || c.startsWith(d.endsWith('/') ? d : `${d}/`)
}

/** Reads the Python version out of a pyvenv.cfg file (venv, virtualenv and uv all write one). */
export const parsePyvenvVersion = (cfg: string): string | undefined => {
  const m = cfg.match(/^\s*(?:version|version_info)\s*=\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)/m)
  return m?.[1]
}

/** True when a venv is active and it is the one for the folder the session works in. */
export const isActiveHere = (f: VenvFacts): boolean => {
  if (!f.virtualEnv) return false
  if (f.projectVenv !== undefined) return trimSlash(f.projectVenv) === trimSlash(f.virtualEnv)
  return isInside(f.cwd, parent(f.virtualEnv))
}

export const describeVenv = (f: VenvFacts): string => {
  const py = (v?: string) => (v ? ` · Python ${v}` : '')

  if (f.virtualEnv) {
    const label = venvLabel(f.virtualEnv)
    const foreign = f.projectVenv !== undefined && trimSlash(f.projectVenv) !== trimSlash(f.virtualEnv)
    if (foreign) {
      return `⚠ Virtual Environment: ${label}${py(f.activeVersion)} · this project has its own ${base(f.projectVenv!)}`
    }
    return `Virtual Environment: ${label}${py(f.activeVersion)}`
  }

  if (f.condaEnv) {
    return `🐍 conda: ${f.condaEnv}${f.projectVenv ? ` · ${base(f.projectVenv)} here is not active` : ''}`
  }

  if (f.projectVenv) {
    return `⚠ no venv active · ${venvLabel(f.projectVenv)} exists but is not activated`
  }

  return `🐍 no venv · system Python${f.systemVersion ? ` ${f.systemVersion}` : ''}`
}
