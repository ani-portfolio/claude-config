#!/usr/bin/env python3
"""Claude Code SessionStart hook for the ML project skills.

Claude Code runs it when a session starts in a project set up by ml-project-setup, with the session's details as
JSON on stdin. It writes `export ML_AGENT_SESSION=<session id>` to the file named by CLAUDE_ENV_FILE, so every
command the session runs has the id. The skills then tag each MLflow run and each git commit with it, which links
a run to the Claude Code trace of the session that made it.

Standard library only. It never fails the session: on any problem it does nothing.
"""

import json
import os
import re
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        return
    session = str(data.get("session_id") or "")
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if not session or not env_file or not re.fullmatch(r"[A-Za-z0-9._:-]+", session):
        return
    try:
        with open(env_file, "a", encoding="utf-8") as f:
            f.write(f"export ML_AGENT_SESSION={session}\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
