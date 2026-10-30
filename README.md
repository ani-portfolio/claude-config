# claude-config

This repo holds my whole Claude Code setup: the global `CLAUDE.md`, settings, hooks, agents, mods, and skills. One clone and one script set it all up on a new machine. The script links `~/.claude` to this repo, so a change made here applies straight away, and `git pull` updates a machine.

## What is here

| Folder | What it holds | Where it is linked |
|---|---|---|
| `config/CLAUDE.md` | My global instructions | `~/.claude/CLAUDE.md` |
| `config/settings.json` | Permissions, hooks, enabled mods, and the model | `~/.claude/settings.json` |
| `hooks/` | The notifier hooks and their config | `~/.claude/hooks` |
| `agents/` | Subagents such as `ml-experimenter` | `~/.claude/agents/<name>.md` |
| `mods/` | The four mods: bg-tasks, side-chat, usage-bar, and venv-status | Installed as plugins |
| `skills/` | The `ml-`, `ab-`, and `help-me-` skills, `grill-me`, `code-map`, their shared code, tools, and tests | `~/.claude/skills/<name>` |

`~/Projects/claude-skills` is also linked to `skills/`. Older projects and some skill text name that path, so it has to keep working.

## Set up a new machine

Install these first: Claude Code, git, Node.js, Python 3, and [uv](https://docs.astral.sh/uv/). The GitHub CLI (`gh`) is the easiest way to clone this private repo.

```bash
gh auth login
mkdir -p ~/Projects
gh repo clone ani-portfolio/claude-config ~/Projects/claude-config
cd ~/Projects/claude-config
./install.sh
```

Then start Claude Code, sign in, and restart it once so it loads the mods. The script does the following:

1. It rewrites the absolute paths in `config/settings.json` to match this machine.
2. It links everything in the table above.
3. It installs the four mods as plugins.

Anything already at those locations moves to `~/.claude/backups/claude-config-<time>/` first, so nothing is lost. The script is safe to run again. If your home folder differs from the last machine, git shows `config/settings.json` as changed, which is expected.

Run this to check the setup. It changes nothing, and it exits with code 1 if something is out of place:

```bash
./install.sh --check
```

Use `./install.sh --no-plugins` to skip the mods.

## What this repo does not hold

These stay out of the repo, and you need to deal with them separately:

* **Logins.** Sign in to Claude Code again on the new machine.
* **Project secrets.** API keys stay in each project's `.env` file and in GitHub secrets.
* **Transcripts and project memory.** These live in `~/.claude/projects/`. They can contain anything you pasted or ran, so they stay out of git. To keep them, copy that folder across once by hand. Its folder names include each project's path, so keep your projects in the same location.
* **Per machine settings.** These go in `settings.local.json` files.
* **Runtime state and caches.** The hooks write state files next to their scripts, and `.gitignore` keeps those out.
* **`skills/_backup/`.** This is a local, ignored folder.

## Start a new project

The skills are installed globally, so a new project needs nothing from this repo. Open the new folder in Claude Code and run one of these:

* `/ml-project-setup` for an ML project. It creates the folders, `.venv`, `ml_spec.yaml`, the project's `CLAUDE.md`, the git repo, and the MLflow experiment. Then run `/ml-status` at any time for the next step.
* `/ab-project-setup` for an A/B test.

The modelling stages log to a shared MLflow server. Start it when it is not running:

```bash
python3 ~/Projects/claude-skills/tools/mlflow_server.py start
```

The conventions behind the skills are in `skills/ML_CONVENTIONS.md` and `skills/AB_CONVENTIONS.md`.

## Change something

Edit the file in this repo, then commit it. Because of the links, the change already applies on this machine, although a running session may need `/reload-skills` or `/reload-plugins`. Other machines get it with `git pull`.

* **Add a skill:** create `skills/<name>/SKILL.md`, then run `./install.sh` to link it. For a new `ml-` or `ab-` skill, follow "Adding a new skill" in the conventions.
* **Remove a skill:** delete its folder, then remove the broken link that `./install.sh --check` reports.
* **Add an agent:** add `agents/<name>.md`, then run `./install.sh`.
* **Change a mod:** edit it in `mods/<name>/`, raise the version in `.claude-plugin/plugin.json`, then run `claude plugin marketplace update <marketplace>` and `claude plugin update <plugin>@<marketplace>`.
* **Change shared skill code:** edit only `skills/_ml_shared/` or `skills/_ab_shared/`, then run the sync tools below.

Claude Code writes to `settings.json` itself, for example when you add a permission, so those changes appear in git. Installing a mod keeps the link in place. If another kind of write ever replaces the link with a plain file, `./install.sh --check` reports it. Copy any changes you want into `config/settings.json`, then run `./install.sh`.

## Run the tests

```bash
python3 -m pytest tests                      # the installer, from the repo root
cd skills
python3 -m pytest tests                      # the skills
python3 tools/sync_ml_shared.py              # shared ML code matches in every skill
python3 tools/sync_ab_shared.py              # shared A/B code matches in every skill
python3 tests/run_regression.py              # the full stages on fixtures, which takes a long time
```

The Python that runs these needs pytest and the packages the skills use, so a project's `.venv/bin/python` is the easiest choice. Add `--fix` to a sync tool to update the copies.
