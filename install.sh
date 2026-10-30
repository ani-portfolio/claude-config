#!/usr/bin/env bash
# Link this repo into ~/.claude, so one clone sets up Claude Code on a new machine.
#
#   ./install.sh               link everything, point the settings at this machine, and install the mods
#   ./install.sh --check       report what is out of place, change nothing, and exit 1 if anything is
#   ./install.sh --no-plugins  skip the mods, which need the claude command
#
# It is safe to run again. A real file or folder in the way moves to ~/.claude/backups/claude-config-<time>/
# first, and an old link is simply replaced.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
BACKUP_DIR="$CLAUDE_DIR/backups/claude-config-$(date +%Y%m%d-%H%M%S)"
CHECK=0
PLUGINS=1
PROBLEMS=0

for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --no-plugins) PLUGINS=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

problem() {
  echo "PROBLEM: $1"
  PROBLEMS=$((PROBLEMS + 1))
}

backup() {
  mkdir -p "$BACKUP_DIR"
  mv "$1" "$BACKUP_DIR/"
  echo "Moved $1 to $BACKUP_DIR"
}

# Make dst a link to src, unless it already is one.
place_link() {
  local src="$1" dst="$2"
  if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$src" ]; then
    return
  fi
  if [ "$CHECK" = 1 ]; then
    problem "$dst is not a link to $src"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  if [ -L "$dst" ]; then
    rm "$dst"
  elif [ -e "$dst" ]; then
    backup "$dst"
  fi
  ln -s "$src" "$dst"
  echo "Linked $dst"
}

# The settings file holds absolute paths to the hooks and the mods. Print "stale" when they do not match this
# machine, and rewrite them unless this is a check.
point_settings() {
  python3 - "$REPO/config/settings.json" "$REPO" "$CLAUDE_DIR" "$CHECK" <<'EOF'
import re
import sys

path, repo, claude, check = sys.argv[1:5]
with open(path, encoding="utf-8") as f:
    old = f.read()
new = re.sub(r'node \\"[^"\\]*/hooks/', lambda m: f'node \\"{claude}/hooks/', old)
new = re.sub(r'("path": ")[^"]*/(?:claude-)?mods/([^"/]+)"', lambda m: f'{m[1]}{repo}/mods/{m[2]}"', new)
if new != old:
    print("stale")
    if check != "1":
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
EOF
}

# Print "<marketplace> <plugin> <registered location>" for each mod, using a dash for a location not yet known.
list_mods() {
  python3 - "$REPO/mods" "$CLAUDE_DIR/plugins/known_marketplaces.json" <<'EOF'
import glob
import json
import os
import sys

mods, known_file = sys.argv[1:3]
known = {}
if os.path.exists(known_file):
    with open(known_file, encoding="utf-8") as f:
        known = json.load(f)
for path in sorted(glob.glob(f"{mods}/*/.claude-plugin/marketplace.json")):
    with open(path, encoding="utf-8") as f:
        market = json.load(f)
    location = known.get(market["name"], {}).get("installLocation", "-")
    for plugin in market["plugins"]:
        print(market["name"], plugin["name"], location)
EOF
}

install_mods() {
  if ! command -v claude >/dev/null 2>&1; then
    problem "the claude command was not found, so the mods were not installed"
    return
  fi
  local market plugin location dir
  while read -r market plugin location; do
    dir="$REPO/mods/$plugin"
    [ "$location" = "$dir" ] && continue
    if [ "$CHECK" = 1 ]; then
      problem "mod $plugin is not installed from $dir"
      continue
    fi
    [ "$location" != "-" ] && claude plugin marketplace remove "$market" >/dev/null 2>&1
    if claude plugin marketplace add "$dir" >/dev/null && claude plugin install "$plugin@$market" --scope user >/dev/null; then
      echo "Installed mod $plugin"
    else
      problem "mod $plugin did not install"
    fi
  done < <(list_mods)
}

for tool in python3 node; do
  command -v "$tool" >/dev/null 2>&1 || problem "$tool was not found"
done

# Settings first: they must be right before Claude Code reads them through the link.
if [ -n "$(point_settings)" ] && [ "$CHECK" = 1 ]; then
  problem "config/settings.json has paths from another machine"
fi

place_link "$REPO/config/CLAUDE.md" "$CLAUDE_DIR/CLAUDE.md"
place_link "$REPO/config/settings.json" "$CLAUDE_DIR/settings.json"
place_link "$REPO/hooks" "$CLAUDE_DIR/hooks"

for file in "$REPO"/agents/*.md; do
  [ -e "$file" ] && place_link "$file" "$CLAUDE_DIR/agents/$(basename "$file")"
done

# Folders starting with an underscore are templates and shared code, not skills.
for file in "$REPO"/skills/*/SKILL.md; do
  dir="$(dirname "$file")"
  name="$(basename "$dir")"
  case "$name" in _*) continue ;; esac
  place_link "$dir" "$CLAUDE_DIR/skills/$name"
done

# Links left behind by a skill or agent that no longer exists.
for link in "$CLAUDE_DIR"/skills/* "$CLAUDE_DIR"/agents/*; do
  if [ -L "$link" ] && [ ! -e "$link" ]; then
    problem "$link is a broken link"
  fi
done

# Older projects and skill text name ~/Projects/claude-skills, so keep that path working.
OLD_PATH="$HOME/Projects/claude-skills"
if [ -e "$OLD_PATH" ] && [ ! -L "$OLD_PATH" ]; then
  if [ "$OLD_PATH" != "$REPO/skills" ]; then
    echo "Left $OLD_PATH alone because it is a real folder, so older projects still use it."
  fi
else
  place_link "$REPO/skills" "$OLD_PATH"
fi

[ "$PLUGINS" = 1 ] && install_mods

if [ "$PROBLEMS" -gt 0 ]; then
  echo "$PROBLEMS problem(s) found."
  [ "$CHECK" = 1 ] && exit 1
  exit 0
fi
echo "Everything is in place."
