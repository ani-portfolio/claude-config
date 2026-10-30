#!/usr/bin/env python3
"""The shared MLflow server for every ML and A/B project on this computer.

One server, one store in your home folder: every project logs to its own experiment in it, and the
experimentation agent logs there too. The server runs only on this computer (127.0.0.1).

    ~/.mlflow/mlflow.db      runs, metrics, parameters, and the model registry (SQLite)
    ~/.mlflow/artifacts/     files logged with the runs: tables, charts, models
    ~/.mlflow/server.log     the server's log

Commands (standard library only, so any Python 3 can run this):
    mlflow_server.py status           is the server up, and where it keeps its data
    mlflow_server.py start            start it in the background (until you log out or stop it)
    mlflow_server.py stop             stop a server started with start
    mlflow_server.py install-agent    macOS: start it at every login and restart it if it stops
    mlflow_server.py uninstall-agent  macOS: remove that login item
    mlflow_server.py url              print the tracking address

MLflow itself is installed once, outside any project, with:  uv tool install mlflow
Settings: MLFLOW_HOME (default ~/.mlflow) and MLFLOW_PORT (default 5000).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("MLFLOW_HOME", "~/.mlflow")).expanduser()
HOST = "127.0.0.1"
PORT = int(os.environ.get("MLFLOW_PORT", "5000"))
URL = f"http://{HOST}:{PORT}"
LABEL = "com.claude-skills.mlflow"
PLIST = Path("~/Library/LaunchAgents").expanduser() / f"{LABEL}.plist"
PID_FILE = HOME / "server.pid"
LOG_FILE = HOME / "server.log"


def is_up(timeout=1.5):
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def _in_venv(path):
    """True for an executable inside a virtual environment, such as a project's .venv: its mlflow is the project's
    client (or an old version uv fell back to), not the server."""
    return (Path(path).resolve().parent.parent / "pyvenv.cfg").exists() or (Path(path).parent.parent / "pyvenv.cfg").exists()


def mlflow_cmd():
    """The mlflow executable: the one from `uv tool install mlflow`, else one on the PATH that is not inside a
    virtual environment, else `uv tool run mlflow`."""
    for cand in (Path("~/.local/bin/mlflow").expanduser(), shutil.which("mlflow")):
        if cand and Path(cand).exists() and (str(cand).startswith(str(Path("~/.local/bin").expanduser()))
                                             or not _in_venv(cand)):
            return [str(cand)]
    uv = shutil.which("uv")
    if uv:
        return [uv, "tool", "run", "mlflow"]
    return None


def server_args():
    cmd = mlflow_cmd()
    if not cmd:
        sys.exit("MLflow is not installed. Install it once with: uv tool install mlflow")
    return cmd + ["server", "--host", HOST, "--port", str(PORT),
                  "--backend-store-uri", f"sqlite:///{HOME / 'mlflow.db'}",
                  "--artifacts-destination", str(HOME / "artifacts")]


def status():
    up = is_up()
    print(f"MLflow server: {'running' if up else 'not running'} at {URL}")
    print(f"Store: {HOME / 'mlflow.db'} ({'exists' if (HOME / 'mlflow.db').exists() else 'not created yet'})")
    print(f"Artifacts: {HOME / 'artifacts'}")
    if sys.platform == "darwin":
        print(f"Starts at login: {'yes' if PLIST.exists() else 'no'} ({PLIST})")
    cmd = mlflow_cmd()
    print("MLflow command: " + (" ".join(cmd) if cmd else "not found; install with: uv tool install mlflow"))
    return 0 if up else 1


def start():
    if is_up():
        print(f"Already running at {URL}")
        return 0
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "artifacts").mkdir(exist_ok=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MLFLOW_DISABLE_TELEMETRY": "true"}
    with open(LOG_FILE, "a") as log:
        p = subprocess.Popen(server_args(), stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
    PID_FILE.write_text(str(p.pid))
    for _ in range(60):
        if is_up():
            print(f"Started at {URL} (process {p.pid}). Log: {LOG_FILE}")
            return 0
        if p.poll() is not None:
            break
        time.sleep(1)
    print(f"The server did not start. Read the end of {LOG_FILE}.")
    return 1


def stop():
    if PLIST.exists():
        print("The server is a login item; it would restart. Run uninstall-agent to stop it for good.")
    if not PID_FILE.exists():
        print("No server started by this script." + (" One is running, started some other way." if is_up() else ""))
        return 0 if not is_up() else 1
    pid = int(PID_FILE.read_text().strip())
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    for _ in range(20):
        if not is_up(0.5):
            print("Stopped.")
            return 0
        time.sleep(0.5)
    print("Asked the server to stop, but it still answers.")
    return 1


def install_agent():
    if sys.platform != "darwin":
        sys.exit("install-agent is for macOS. On Linux, run `start` from your shell profile or a systemd user unit.")
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "artifacts").mkdir(exist_ok=True)
    args = "\n".join(f"    <string>{a}</string>" for a in server_args())
    path_env = os.pathsep.join([str(Path('~/.local/bin').expanduser()), "/opt/homebrew/bin", "/usr/local/bin",
                                "/usr/bin", "/bin"])
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>{path_env}</string>
    <key>MLFLOW_DISABLE_AGENT_HINT</key><string>1</string>
    <key>MLFLOW_DISABLE_TELEMETRY</key><string>true</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{LOG_FILE}</string>
  <key>StandardErrorPath</key><string>{LOG_FILE}</string>
</dict>
</plist>
""")
    if PID_FILE.exists():
        stop()
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LABEL}"], capture_output=True)
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"launchctl could not load {PLIST}: {r.stderr.strip()}")
        return 1
    for _ in range(60):
        if is_up():
            print(f"Installed. The server runs at {URL} now and starts at every login.")
            return 0
        time.sleep(1)
    print(f"Installed, but the server is not answering yet. Read {LOG_FILE}.")
    return 1


def uninstall_agent():
    if sys.platform != "darwin":
        sys.exit("uninstall-agent is for macOS.")
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    PLIST.unlink(missing_ok=True)
    print("Removed the login item. The data in " + str(HOME) + " is kept.")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    actions = {"status": status, "start": start, "stop": stop, "install-agent": install_agent,
               "uninstall-agent": uninstall_agent, "url": lambda: print(URL) or 0}
    if cmd not in actions:
        sys.exit(__doc__)
    sys.exit(actions[cmd]())


if __name__ == "__main__":
    main()
