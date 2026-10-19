"""Git for the projects of the ML project skills.

Every project is a git repository. Work never lands on main: main holds only an empty first commit, and each
version of the model (V1, V2, ...) has its own branch, ml/v1, ml/v2, ... branched from the version before it. A
person merges a version into main when they want to; the skills never do.

    setup                one empty commit on main, then the ml/v1 branch with the set up files
    each stage build     one commit on the version branch with the stage's notebook, the spec, the kept answers,
                         and the project's code and documents (data and models are ignored by .gitignore)
    a new version        ml-iterate tags the previous version (v1) and branches ml/v2 from it
    closing a version    kept or stopped: tag v2; rolled back: tag v2-rolled-back and switch back to ml/v1

Only named paths are committed, so a person's other changes stay out of the skills' commits. Every function here
is safe to call in a folder without git: it does nothing and says why.

Command line (from the project root):
    git_ops.py status         the branch, the version branch, the last commit, and uncommitted files
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.path.exists(os.path.join(HERE, "ab_common.py")):
    import ab_common as ml_common  # noqa: E402
else:
    import ml_common  # noqa: E402

PROTECTED = ("main", "master")
TRAILER = "Committed-By: ml project skills"


def git(root, *args, check=False):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
    return r


def available():
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


def is_repo(root):
    return available() and git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


def identity(root):
    """(name, email) git will commit with, or None when either is missing."""
    name = git(root, "config", "user.name").stdout.strip()
    email = git(root, "config", "user.email").stdout.strip()
    return (name, email) if name and email else None


def branch(root):
    r = git(root, "branch", "--show-current")
    return r.stdout.strip() or None


def head(root):
    r = git(root, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def branch_exists(root, name):
    return git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{name}").returncode == 0


def tag_exists(root, name):
    return git(root, "show-ref", "--verify", "--quiet", f"refs/tags/{name}").returncode == 0


def version_number(spec):
    v = str(((spec or {}).get("version") or {}).get("current") or "V1")
    m = re.match(r"[Vv](\d+)", v)
    return int(m.group(1)) if m else 1


def version_branch(spec, number=None):
    return f"ml/v{number or version_number(spec)}"


def init_repo(root, spec=None):
    """A new repository with an empty first commit on main and the version branch checked out. In an existing
    repository, only moves off main to the version branch. Returns (done, message)."""
    if not available():
        return False, "git is not installed, so the project is not a git repository."
    target = version_branch(spec)
    if is_repo(root):
        b = branch(root)
        if b in PROTECTED or b is None:
            if head(root) is None:   # a repository with no commits yet: start main with the empty commit
                if not identity(root):
                    return False, _no_identity()
                git(root, "commit", "--allow-empty", "-m", "Initialize repository", check=True)
            git(root, "switch", target) if branch_exists(root, target) else git(root, "switch", "-c", target, check=True)
            return True, f"existing git repository: switched from {b or 'no branch'} to {target}"
        return True, f"existing git repository on branch {b}"
    if not identity(root):
        return False, _no_identity()
    git(root, "init", "-q", "-b", "main", check=True)
    git(root, "commit", "-q", "--allow-empty", "-m", "Initialize repository", check=True)
    git(root, "switch", "-q", "-c", target, check=True)
    return True, f"created a git repository: an empty first commit on main, work on {target}"


def _no_identity():
    return ("git has no user name or email, so nothing was committed. Set them once with "
            "`git config --global user.name \"Your Name\"` and `git config --global user.email you@example.com`.")


def ensure_work_branch(root, spec=None):
    """Make sure commits go to a branch other than main. On main (or with no branch), switch to the version
    branch, creating it from the current commit. Any other branch a person chose is kept."""
    b = branch(root)
    if b not in PROTECTED and b is not None:
        return b
    target = version_branch(spec)
    if branch_exists(root, target):
        git(root, "switch", "-q", target, check=True)
    else:
        git(root, "switch", "-q", "-c", target, check=True)
    return target


def commit_paths(root, paths, message, spec=None, trailers=None):
    """Commit only the given paths (files or folders, ignored files left out) on a work branch. Returns the new
    commit id, or None when nothing changed or git is not usable here."""
    if not is_repo(root) or not identity(root):
        return None
    ensure_work_branch(root, spec)
    rels = []
    for p in paths:
        if not p:
            continue
        rel = os.path.relpath(p, root) if os.path.isabs(p) else p
        if os.path.exists(os.path.join(root, rel)) or git(root, "ls-files", "--error-unmatch", rel).returncode == 0:
            rels.append(rel)
    # Ignored files named directly (a restored model, say) would make `git add` refuse; leave them out
    ignored = set(git(root, "check-ignore", "--", *rels).stdout.splitlines()) if rels else set()
    rels = [r for r in rels if r not in ignored]
    if not rels:
        return None
    git(root, "add", "--all", "--", *rels)
    # Commit exactly the files that changed under those paths (an empty folder is not a path git knows)
    files = [f for f in git(root, "diff", "--cached", "--name-only", "--no-renames", "--", *rels).stdout.splitlines() if f]
    if not files:
        return None
    body = message + "\n\n" + "\n".join([f"{k}: {v}" for k, v in (trailers or {}).items() if v] + [TRAILER])
    r = git(root, "commit", "-q", "-m", body, "--", *files)
    if r.returncode != 0:
        print(f"git: the commit failed: {(r.stderr or r.stdout).strip()[:300]}", file=sys.stderr)
        return None
    return head(root)


def tag(root, name, message):
    if not is_repo(root) or tag_exists(root, name):
        return False
    return git(root, "tag", "-a", name, "-m", message).returncode == 0


def stage_paths(spec, root, stage, spec_path):
    """What a stage build commits: the spec, the stage notebook, its kept answers, and the project's code and
    documents. Data and models stay out through .gitignore."""
    paths = ml_common.project_paths(spec, root)
    nb = ((spec.get("notebooks") or {}).get(stage) or {}).get("path")
    number = getattr(ml_common, "STAGE_NUMBER", {}).get(stage)
    answers = os.path.join(root, "answers", (f"{number:02d}_" if isinstance(number, int) else "") + f"{stage}.yaml")
    return [spec_path, os.path.join(root, nb) if nb else None, answers, paths["src"], paths["docs"],
            os.path.join(root, ".gitignore")]


# ---------------------------------------------------------------- worktrees for experiments

def worktree_add(root, path, new_branch, start="HEAD"):
    """A new worktree at path on a new branch made from start. ml-autopilot runs each experiment in one, so the
    project itself is never touched by an experiment. Returns (done, message)."""
    if not is_repo(root):
        return False, "the project is not a git repository, so no worktree was made"
    if new_branch in PROTECTED:
        return False, f"refusing to make a worktree on {new_branch}"
    if branch_exists(root, new_branch):
        return False, f"the branch {new_branch} exists already"
    r = git(root, "worktree", "add", "-q", "-b", new_branch, path, start)
    if r.returncode != 0:
        return False, f"git worktree add failed: {(r.stderr or r.stdout).strip()[:300]}"
    return True, f"worktree {path} on branch {new_branch}"


def worktree_remove(root, path):
    """Remove a worktree, even with untracked or ignored files in it (the data and model copies). Its branch is
    kept for the record."""
    if not is_repo(root):
        return False
    r = git(root, "worktree", "remove", "--force", path)
    git(root, "worktree", "prune")
    return r.returncode == 0


def worktrees(root):
    """[{path, branch, head}] for every worktree of the repository, the main one first."""
    if not is_repo(root):
        return []
    out, cur = [], {}
    for line in git(root, "worktree", "list", "--porcelain").stdout.splitlines():
        if not line.strip():
            if cur:
                out.append(cur)
            cur = {}
            continue
        key, _, val = line.partition(" ")
        if key == "worktree":
            cur["path"] = val
        elif key == "branch":
            cur["branch"] = val.removeprefix("refs/heads/")
        elif key == "HEAD":
            cur["head"] = val
    if cur:
        out.append(cur)
    return out


def merge_branch(root, source, message, prefer=None):
    """Merge source into the current branch with a merge commit. Never on main. prefer='theirs' settles conflicting
    hunks in favour of source (ml-autopilot adopting an experiment, whose spec is the one to keep). Returns
    (done, message)."""
    if not is_repo(root):
        return False, "the project is not a git repository"
    b = branch(root)
    if b in PROTECTED or b is None:
        return False, f"refusing to merge into {b or 'no branch'}"
    if not identity(root):
        return False, _no_identity()
    r = git(root, "merge", "--no-ff", "-q", *(["-X", prefer] if prefer else []), "-m", f"{message}\n\n{TRAILER}", source)
    if r.returncode != 0:
        git(root, "merge", "--abort")
        return False, f"git merge failed and was aborted: {(r.stderr or r.stdout).strip()[:300]}"
    return True, f"merged {source} into {b}"


def status(root):
    if not is_repo(root):
        return {"repo": False}
    porcelain = git(root, "status", "--porcelain").stdout.splitlines()
    last = git(root, "log", "-1", "--format=%h %s").stdout.strip()
    return {"repo": True, "branch": branch(root), "head": head(root), "last_commit": last,
            "uncommitted": [l[3:] for l in porcelain][:50], "on_protected": branch(root) in PROTECTED}


def main():
    root = ml_common.find_project_root()[0] or "."
    import json
    print(json.dumps(status(root), indent=1))


if __name__ == "__main__":
    main()
