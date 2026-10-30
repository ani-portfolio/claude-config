"""Test install.sh through its command line, on a small fake repo and a throwaway home folder."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "install.sh"
OLD_SETTINGS = {
    "hooks": {"Stop": [{"hooks": [{"type": "command",
                                   "command": 'node "/Users/someone/.claude/hooks/on-stop.js"'}]}]},
    "extraKnownMarketplaces": {"demo": {"source": {"source": "directory",
                                                   "path": "/Users/someone/Projects/claude-mods/demo"}}},
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "work" / "claude-config"
    (root / "config").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "install.sh")
    (root / "config" / "CLAUDE.md").write_text("# rules\n")
    (root / "config" / "settings.json").write_text(json.dumps(OLD_SETTINGS, indent=2))
    (root / "hooks").mkdir()
    (root / "hooks" / "on-stop.js").write_text("// hook\n")
    (root / "agents").mkdir()
    (root / "agents" / "helper.md").write_text("agent\n")
    for skill in ("ml-one", "grill-me", "_ml-skill-template"):
        (root / "skills" / skill).mkdir(parents=True)
        (root / "skills" / skill / "SKILL.md").write_text("skill\n")
    (root / "skills" / "tools").mkdir()
    (root / "mods" / "demo" / ".claude-plugin").mkdir(parents=True)
    return root


@pytest.fixture
def home(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


def run(repo, home, *args):
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CONFIG_DIR"}
    env["HOME"] = str(home)
    return subprocess.run(["bash", str(repo / "install.sh"), "--no-plugins", *args], env=env, text=True,
                          capture_output=True)


def links_to(path, target):
    return path.is_symlink() and path.resolve() == target.resolve()


def backups(home):
    return sorted((home / ".claude" / "backups").glob("claude-config-*/*"))


def test_fresh_home_links_everything(repo, home):
    result = run(repo, home)
    assert result.returncode == 0, result.stderr
    claude = home / ".claude"
    assert links_to(claude / "CLAUDE.md", repo / "config" / "CLAUDE.md")
    assert links_to(claude / "settings.json", repo / "config" / "settings.json")
    assert links_to(claude / "hooks", repo / "hooks")
    assert links_to(claude / "agents" / "helper.md", repo / "agents" / "helper.md")
    assert links_to(claude / "skills" / "ml-one", repo / "skills" / "ml-one")
    assert links_to(claude / "skills" / "grill-me", repo / "skills" / "grill-me")
    assert not (home / "Projects" / "claude-skills").exists()


def test_only_real_skills_are_linked(repo, home):
    run(repo, home)
    names = sorted(p.name for p in (home / ".claude" / "skills").iterdir())
    assert names == ["grill-me", "ml-one"]


def test_settings_are_pointed_at_this_machine(repo, home):
    run(repo, home)
    settings = json.loads((repo / "config" / "settings.json").read_text())
    command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert command == f'node "{home}/.claude/hooks/on-stop.js"'
    path = settings["extraKnownMarketplaces"]["demo"]["source"]["path"]
    assert path == str(repo / "mods" / "demo")


def test_second_run_changes_nothing(repo, home):
    run(repo, home)
    settings = (repo / "config" / "settings.json").read_text()
    result = run(repo, home)
    assert result.returncode == 0, result.stderr
    assert backups(home) == []
    assert (repo / "config" / "settings.json").read_text() == settings


def test_existing_files_are_backed_up_not_lost(repo, home):
    claude = home / ".claude"
    (claude / "hooks").mkdir(parents=True)
    (claude / "hooks" / "mine.js").write_text("mine\n")
    (claude / "CLAUDE.md").write_text("my old rules\n")
    run(repo, home)
    saved = {p.name: p for p in backups(home)}
    assert saved["CLAUDE.md"].read_text() == "my old rules\n"
    assert (saved["hooks"] / "mine.js").read_text() == "mine\n"
    assert links_to(claude / "CLAUDE.md", repo / "config" / "CLAUDE.md")


def test_old_links_are_replaced_without_a_backup(repo, home, tmp_path):
    old = tmp_path / "old" / "ml-one"
    old.mkdir(parents=True)
    skills = home / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "ml-one").symlink_to(old)
    run(repo, home)
    assert links_to(skills / "ml-one", repo / "skills" / "ml-one")
    assert backups(home) == []


def test_other_folders_in_skills_are_left_alone(repo, home):
    synced = home / ".claude" / "skills" / "synced"
    synced.mkdir(parents=True)
    (synced / "file").write_text("keep\n")
    run(repo, home)
    assert (synced / "file").read_text() == "keep\n"
    assert not synced.is_symlink()


def test_check_passes_after_install_and_changes_nothing(repo, home):
    run(repo, home)
    settings = (repo / "config" / "settings.json").read_text()
    result = run(repo, home, "--check")
    assert result.returncode == 0, result.stdout
    assert (repo / "config" / "settings.json").read_text() == settings


def test_check_on_an_empty_home_fails_and_creates_nothing(repo, home):
    result = run(repo, home, "--check")
    assert result.returncode == 1
    assert list(home.iterdir()) == []


def test_check_reports_a_link_replaced_by_a_file(repo, home):
    run(repo, home)
    settings = home / ".claude" / "settings.json"
    settings.unlink()
    settings.write_text("{}\n")
    result = run(repo, home, "--check")
    assert result.returncode == 1
    assert "settings.json" in result.stdout


def test_check_reports_a_broken_skill_link(repo, home, tmp_path):
    run(repo, home)
    (home / ".claude" / "skills" / "gone").symlink_to(tmp_path / "missing")
    result = run(repo, home, "--check")
    assert result.returncode == 1
    assert "gone" in result.stdout


def test_script_can_be_run_directly():
    assert os.access(SCRIPT, os.X_OK)
