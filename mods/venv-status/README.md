# venv-status

A Claude Code mod that shows, in the band above the prompt, which Python virtual environment the session is using. It shares that band (and its frame) with the other mods that draw there, such as usage-bar and bg-tasks.

## What it shows

| Situation | Line above the prompt |
| --- | --- |
| The project's own venv is active ("Virtual Environment" in green) | `Virtual Environment: martech_adtech_data/.venv · Python 3.11.6` |
| A different venv is active than the project's own | `⚠ Virtual Environment: flinks_practice/.venv · Python 3.10.13 · this project has its own .venv` |
| The project has a venv, but none is active | `⚠ no venv active · martech_adtech_data/.venv exists but is not activated` |
| A conda environment is active | `🐍 conda: base` |
| No venv anywhere | `🐍 no venv · system Python 3.11.6` |

The line updates when the session starts, when the working directory changes, and after every turn.

## VS Code and the desktop app

Where the band above the prompt is not drawn, the mod still reaches you:

1. Type `/venv` to print the line in the conversation.
2. When the line is a warning (⚠), it is shown beneath Claude's answer, once each time the warning changes.

Warnings (⚠) are drawn in yellow in the band.

## How it decides

1. The active venv is read from `VIRTUAL_ENV` (and `CONDA_DEFAULT_ENV` for conda). Bash commands that Claude runs inherit this environment.
2. The project venv is a `.venv`, `venv` or `env` folder in the working directory that holds a `pyvenv.cfg` file.
3. The Python version comes from the venv's `pyvenv.cfg`. When no venv exists at all, it asks `python3` for its version.

## Install

From a Claude Code session on this Mac:

```
/plugin install venv-status --marketplace ~/Projects/claude-mods/venv-status
```

Answer `y` to add the marketplace, then pick the user scope so it runs in every session.

To try it once without installing:

```
claude --plugin-dir ~/Projects/claude-mods/venv-status
```

## Tip

Start Claude Code after activating the project's venv (`source .venv/bin/activate && claude`), so every Bash command Claude runs uses that environment.

## Development

```
claude plugin validate ~/Projects/claude-mods/venv-status
claude plugin test ~/Projects/claude-mods/venv-status
```
