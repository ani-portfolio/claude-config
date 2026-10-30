#!/usr/bin/env python3
"""Map a Python repo's src/ as an interactive page: groups and the store and registry names they read and write,
files and the imports between them with the names used, every caller of each shared file, and tours that read the
code depth first from its entry points. Plain parsing with the standard library, so a run costs no tokens.

    python3 code_map.py <repo_root> [--ml | --no-ml] [--out PATH] [--json] [--open]
    python3 code_map.py detect <repo_root>
    python3 code_map.py setup <repo_root> --ml | --no-ml

The map prints the page path, then one "warning:" line per assumption the repo does not meet. It exits with code 1
only when the repo has no src/ folder.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tomllib
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = ".code_map"
SHARED = "shared"
SKIP_DIRS = {"tests", "__pycache__"}
WRITE_CALLS = ("create", "get_or_create", "save")
READ_CALLS = ("get", "read")
# The platform a store name lives on, from words in the calls that read or write it; the first match wins.
PLATFORMS = (("Feature store", ("feature_group", "feature_view", "feature_store")), ("Model registry", ("model",)),
             ("Serving", ("deployment", "serving", "endpoint")), ("Jobs", ("job",)))
TASK_LABEL = "Code map"


# Files and imports

def _source_files(src: Path) -> list[Path]:
    def keep(p):
        parts = p.relative_to(src).parts
        return not SKIP_DIRS & set(parts[:-1]) and not any(s.startswith(".") for s in parts) \
            and not p.name.startswith("test_")
    return sorted(p for p in src.rglob("*.py") if keep(p))


def _module_names(src: Path, path: Path) -> list[str]:
    """The dotted names a file is imported by: relative to src/, and with src. in front for repos where src is a
    package (from src.features import x)."""
    parts = list(path.relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    name = ".".join(parts)
    return [n for n in (name, ".".join(["src", *parts])) if n]


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


class _Scan:
    """One parsed file: its imports of other src files, the local names they bind, and its module constants."""

    def __init__(self, fid, path, tree, src, index):
        self.id, self.path, self.tree = fid, path, tree
        self.doc = next(iter((ast.get_docstring(tree) or "").strip().splitlines()), "")
        self.targets: list[str] = []            # imported src files, in order of first import
        self.names: dict[str, set] = {}         # imported file -> names used from it
        self.module_alias: dict[str, str] = {}  # local dotted name -> src file it stands for
        self.name_from: dict[str, tuple] = {}   # local name -> (src file, original name)
        package = list(path.relative_to(src).parts[:-1])
        imports = sorted((n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
                         key=lambda n: (n.lineno, n.col_offset))
        for node in imports:
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in index:
                        self._link(index[a.name])
                        self.module_alias[a.asname or a.name] = index[a.name]
                continue
            base = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            for a in node.names:
                sub = f"{module}.{a.name}" if module else a.name
                if sub in index:
                    self._link(index[sub])
                    self.module_alias[a.asname or a.name] = index[sub]
                elif module in index:
                    self._link(index[module], a.name)
                    self.name_from[a.asname or a.name] = (index[module], a.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _dotted(node.value) in self.module_alias:
                self.names[self.module_alias[_dotted(node.value)]].add(node.attr)
        self.constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id.isupper():
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue
                if isinstance(value, (str, tuple)):
                    self.constants[node.targets[0].id] = value
        self.main = any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                        and _dotted(n.test.left) == "__name__" for n in tree.body) or path.name == "__main__.py"

    def _link(self, target, name=None):
        if target == self.id:
            return
        if target not in self.names:
            self.targets.append(target)
            self.names[target] = set()
        if name:
            self.names[target].add(name)


# Grouping

def _groups_root(src: Path) -> Path:
    """Descend while a folder holds exactly one package and no loose modules, so src/<pkg>/ groups by subpackage."""
    d = src
    while True:
        packages = [c for c in d.iterdir() if c.is_dir() and (c / "__init__.py").exists() and c.name not in SKIP_DIRS]
        loose = [f for f in d.glob("*.py") if f.name != "__init__.py"]
        if len(packages) != 1 or loose:
            return d
        d = packages[0]


def _group_of(path: Path, root: Path, packages: set) -> str:
    try:
        first = path.relative_to(root).parts[0]
    except ValueError:
        return SHARED
    return first if first in packages else SHARED


# Store and registry names

def _call_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


def _call_kind(call: ast.Call) -> str:
    name = _call_name(call)
    if name.startswith(WRITE_CALLS):
        return "write"
    # A bare get( is almost always dict.get on a column or tag name; store clients name what they get.
    return "read" if name.startswith(READ_CALLS) and name != "get" else "touch"


def _no_calls(node):
    """Every node under an argument, without going into calls nested in it, which are scanned on their own."""
    if isinstance(node, ast.Call):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _no_calls(child)


def _constant_uses(scan: _Scan, scans: dict):
    """(src file, constant, kind, defined, line, call name) for every upper case name of a src module passed to a
    call, directly or through a local name bound to it (name, version = platform.FEATURES). defined is False when
    the module has no literal constant of that name."""
    def ref(node):
        if isinstance(node, ast.Attribute) and _dotted(node.value) in scan.module_alias and node.attr.isupper():
            return scan.module_alias[_dotted(node.value)], node.attr
        if isinstance(node, ast.Name) and node.id in scan.name_from and node.id.isupper():
            return scan.name_from[node.id]
        if isinstance(node, ast.Name) and node.id in scan.constants:
            return scan.id, node.id
        return None

    parents = {c: p for p in ast.walk(scan.tree) for c in ast.iter_child_nodes(p)}
    bound: dict = {}  # (function node, local name) -> constant ref
    for node in ast.walk(scan.tree):
        if isinstance(node, ast.Assign) and (r := ref(node.value)):
            scope = _scope(node, parents)
            for target in node.targets:
                names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                for t in names:
                    if isinstance(t, ast.Name):
                        bound[scope, t.id] = r
    for call in (n for n in ast.walk(scan.tree) if isinstance(n, ast.Call)):
        scope, kind = _scope(call, parents), _call_kind(call)
        for arg in [*call.args, *(k.value for k in call.keywords)]:
            for node in _no_calls(arg):
                r = ref(node) or (isinstance(node, ast.Name) and bound.get((scope, node.id)))
                if r:
                    yield r[0], r[1], kind, r[1] in scans[r[0]].constants, node.lineno, _call_name(call)


def _scope(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _assets(scans: dict):
    """The constants some call reads or writes, as {(module file, name): platform or None}, every use of them
    deduplicated per file, and the names that read or write calls take from the same modules but that are not
    literal constants there."""
    uses = [(s.id, *u) for s in scans.values() for u in _constant_uses(s, scans)]
    calls = {}
    for _, f, c, kind, defined, _, call in uses:
        if defined and kind != "touch":
            calls.setdefault((f, c), []).append(call.lower())
    stored = {fc: next((p for p, words in PLATFORMS if any(w in n for n in names for w in words)), None)
              for fc, names in calls.items()}
    seen, edges = set(), []
    for fid, f, c, kind, _, line, _ in uses:
        if (f, c) in stored and (fid, f, c, kind) not in seen:
            seen.add((fid, f, c, kind))
            edges.append((fid, f, c, kind, line))
    modules = {f for f, _ in stored}
    unresolved = sorted({(fid, f, c, line) for fid, f, c, kind, defined, line, _ in uses
                         if not defined and kind != "touch" and f in modules})
    return stored, edges, unresolved


# Columns: a left to right layout, like the ml-system-design diagram but read from the code

def _rank(nodes, edges: dict):
    """Columns for a weighted directed graph: {node: rank}, where a rank is the longest path from a node with no
    incoming edges, and the edges removed to break loops. Self loops are dropped. Each loop loses its weakest edge;
    on a tie, the one that jumps furthest backwards in steps from the start nodes."""
    nodes = sorted(nodes)
    edges = {(a, b): w for (a, b), w in edges.items() if a != b and a in nodes and b in nodes}
    removed = []
    while cycle := _find_cycle(nodes, edges):
        hops = _hops(nodes, edges)
        back = min(cycle, key=lambda e: (edges[e], hops[e[1]] - hops[e[0]], e))
        del edges[back]
        removed.append(back)
    rank = {}
    for n in _topo(nodes, edges):
        rank[n] = max((rank[a] + 1 for a, b in edges if b == n), default=0)
    return rank, removed


def _find_cycle(nodes, edges):
    out = {n: sorted(b for a, b in edges if a == n) for n in nodes}
    state, stack = {}, []

    def visit(n):
        state[n] = "open"
        stack.append(n)
        for m in out[n]:
            if state.get(m) == "open":
                loop = stack[stack.index(m):] + [m]
                return list(zip(loop, loop[1:]))
            if m not in state and (found := visit(m)):
                return found
        state[n] = "done"
        stack.pop()
        return None
    for n in nodes:
        if n not in state and (found := visit(n)):
            return found
    return None


def _hops(nodes, edges):
    """Fewest steps from a node with no incoming edges (the first node by name when every node has one)."""
    targets = {b for _, b in edges}
    frontier = [n for n in nodes if n not in targets] or nodes[:1]
    hops, level = dict.fromkeys(frontier, 0), 0
    while frontier:
        level += 1
        frontier = sorted({b for a, b in edges if a in frontier and b not in hops})
        hops.update(dict.fromkeys(frontier, level))
    return {n: hops.get(n, len(nodes)) for n in nodes}


def _topo(nodes, edges):
    indeg = {n: sum(1 for _, b in edges if b == n) for n in nodes}
    ready, order = [n for n in nodes if not indeg[n]], []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for a, b in sorted(edges):
            if a == n:
                indeg[b] -= 1
                if not indeg[b]:
                    ready.append(b)
    return order


def _group_cols(groups: list[str], edges: dict, step: int) -> dict:
    """Ranked groups at step x rank; groups with no edges in one last column."""
    linked = sorted({n for e in edges for n in e if e[0] != e[1]})
    rank, _ = _rank(linked, edges)
    last = (max(rank.values()) + 1) * step if rank else 0
    return {g: rank[g] * step if g in rank else last for g in groups}


def _store_cols(group, imports, asset_edges, ranked):
    """Group and asset columns from the store arrows. A shared file's reads and writes count for every group that
    imports it, directly or through other shared files."""
    reach = {}  # shared file -> groups that reach it
    for g in ranked:
        frontier = {e["to"] for e in imports if group[e["from"]] == g and group[e["to"]] == SHARED}
        seen = set()
        while frontier:
            seen |= frontier
            frontier = {e["to"] for e in imports if e["from"] in frontier and group[e["to"]] == SHARED} - seen
        for f in seen:
            reach.setdefault(f, set()).add(g)
    writers, readers = {}, {}
    for e in asset_edges:
        if e["dir"] == "touch":
            continue
        by = reach.get(e["file"], set()) if group[e["file"]] == SHARED else {e["group"]}
        (writers if e["dir"] == "write" else readers).setdefault(e["asset"], set()).update(by)
    edges = {}
    for a in writers:
        for w in writers[a]:
            for r in readers.get(a, ()):
                edges[w, r] = edges.get((w, r), 0) + 1
    gcol = _group_cols(ranked, edges, 2)
    acol = {}
    for a in {e["asset"] for e in asset_edges}:
        if writers.get(a):
            acol[a] = min(gcol[g] for g in writers[a]) + 1
        elif readers.get(a):
            acol[a] = min(gcol[g] for g in readers[a]) - 1
        else:
            acol[a] = max(gcol.values(), default=0) + 1
    shift = -min([0, *acol.values()])
    return {g: c + shift for g, c in gcol.items()}, {a: c + shift for a, c in acol.items()}


def _file_cols(files, group, imports, entries) -> dict:
    """Each file's column inside its own group: depth from the group's entry files and the files nothing else in
    the group imports."""
    cols = {}
    for g in set(group.values()):
        mine = [f for f in files if group[f] == g]
        edges = {(e["from"], e["to"]): 1 for e in imports
                 if group[e["from"]] == g == group[e["to"]] and e["to"] not in entries}
        cols.update(_rank(mine, edges)[0])
    return cols


# The model

def _scan_repo(root: Path):
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"{root} has no src/ folder")
    paths = _source_files(src)
    index = {name: f"src/{p.relative_to(src).as_posix()}" for p in paths for name in _module_names(src, p)}
    scans, warnings = {}, []
    for p in paths:
        fid = f"src/{p.relative_to(src).as_posix()}"
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except (SyntaxError, UnicodeDecodeError) as e:
            warnings.append(f"{fid} could not be parsed and is left out: {e}")
            continue
        scans[fid] = (p, tree)
    index = {n: f for n, f in index.items() if f in scans}
    scans = {fid: _Scan(fid, p, tree, src, index) for fid, (p, tree) in scans.items()}
    root_dir = _groups_root(src)
    packages = {c.name for c in root_dir.iterdir() if c.is_dir() and (c / "__init__.py").exists()
                and c.name not in SKIP_DIRS}
    return src, scans, root_dir, packages, index, warnings


def detect(root) -> dict:
    """What suggests the repo follows the ML deployment layout: *_pipeline packages, and constants that calls read
    or write (feature groups, feature views, models)."""
    _, scans, _, packages, _, _ = _scan_repo(Path(root).resolve())
    stored, _, _ = _assets(scans)
    pipelines = sorted(p for p in packages if p.endswith("_pipeline"))
    assets = sorted({c for _, c in stored})
    return {"pipelines": pipelines, "assets": assets, "suggest_ml": bool(pipelines) or len(assets) >= 2}


def _read_config(root: Path) -> dict:
    try:
        return json.loads((root / OUT_DIR / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def build_model(root, ml: bool | None = None) -> dict:
    root = Path(root).resolve()
    if ml is None:
        ml = _read_config(root).get("ml")
    if ml is None:
        ml = detect(root)["suggest_ml"]
    src, scans, root_dir, packages, index, warnings = _scan_repo(root)
    group = {fid: _group_of(s.path, root_dir, packages) for fid, s in scans.items()}
    groups = [{"id": p, "kind": "group"} for p in sorted(packages) if p in group.values()]
    groups += [{"id": SHARED, "kind": "shared"}] if SHARED in group.values() else []
    files = [{"id": fid, "group": group[fid], "doc": s.doc, "path": str(s.path.resolve())} for fid, s in scans.items()]
    imports = [{"from": fid, "to": t, "names": sorted(s.names[t])} for fid, s in scans.items() for t in s.targets]

    callers: dict = {}
    for e in imports:
        if group[e["to"]] == SHARED:
            for name in e["names"] or ["(module)"]:
                callers.setdefault(e["to"], {}).setdefault(name, []).append(e["from"])
    callers = {f: {n: sorted(c) for n, c in sorted(by.items())} for f, by in sorted(callers.items())}

    assets, asset_edges = [], []
    if ml:
        stored, edges, unresolved = _assets(scans)
        clash = {c for _, c in stored if sum(1 for _, c2 in stored if c2 == c) > 1}
        aid = {(f, c): (f"{Path(f).stem}.{c}" if c in clash else c) for f, c in stored}
        for f, c in sorted(stored, key=lambda fc: aid[fc]):
            value = scans[f].constants[c]
            assets.append({"id": aid[f, c], "via": Path(f).stem, "platform": stored[f, c],
                           "value": value if isinstance(value, str) else " ".join(map(str, value))})
        asset_edges = [{"group": group[fid], "asset": aid[f, c], "dir": kind, "file": fid, "line": line}
                       for fid, f, c, kind, line in edges]
        warnings += [f"{Path(f).stem}.{c} in {fid}:{line} is not a constant of {f}, so it is not on the map"
                     for fid, f, c, line in unresolved]
        if not assets:
            warnings.append("ML mode is on but no store or registry names were found in read or write calls")

    tours = [{"entry": e, "steps": _walk(e, scans)} for e in _entries(root, src, scans, index, root_dir, ml)]
    if not tours:
        warnings.append("no entry points found (no jobs/, [project.scripts], __main__ blocks, or package "
                        "__init__.py), so there is no tour")
    ranked = [g["id"] for g in groups if g["kind"] == "group"]
    if ml:
        gcol, acol = _store_cols(group, imports, asset_edges, ranked)
    else:
        between = {}
        for e in imports:
            a, b = group[e["from"]], group[e["to"]]
            if SHARED not in (a, b):
                between[a, b] = between.get((a, b), 0) + 1
        gcol, acol = _group_cols(ranked, between, 1), {}
    entries = [t["entry"] for t in tours]
    for g in groups:
        g["col"] = gcol.get(g["id"])
        init = f"src/{(root_dir / g['id'] / '__init__.py').relative_to(src).as_posix()}"
        g["doc"] = scans[init].doc if g["kind"] == "group" and init in scans else ""
        g["jobs"] = [e for e in entries if group[e] == g["id"]]
        if all(Path(f["id"]).name == "__init__.py" for f in files if f["group"] == g["id"]) and g["kind"] == "group":
            g["empty"] = True
    for a in assets:
        a["col"] = acol[a["id"]]
    fcol = _file_cols([f["id"] for f in files], group, imports, set(entries))
    for f in files:
        f["col"] = fcol[f["id"]]

    missing = [fid for fid, s in scans.items() if not s.doc]
    if missing:
        warnings.append(f"{len(missing)} files have no module docstring, so the tour shows only their names: "
                        + ", ".join(missing))
    return {"root": str(root), "ml": bool(ml), "groups": groups, "files": files, "imports": imports,
            "assets": assets, "asset_edges": asset_edges, "callers": callers, "tours": tours, "warnings": warnings}


def _entries(root, src, scans, index, root_dir, ml) -> list[str]:
    if ml:
        jobs = [fid for fid in scans if "jobs" in Path(fid).parts[:-1]]
        if jobs:
            return sorted(jobs)
    entries = []
    try:
        scripts = tomllib.loads((root / "pyproject.toml").read_text()).get("project", {}).get("scripts", {})
    except (OSError, tomllib.TOMLDecodeError):
        scripts = {}
    for target in scripts.values():
        module = target.split(":")[0].strip()
        if module in index and index[module] not in entries:
            entries.append(index[module])
    entries += [fid for fid, s in sorted(scans.items()) if s.main and fid not in entries]
    if not entries:
        init = f"src/{(root_dir / '__init__.py').relative_to(src).as_posix()}"
        if init in scans:
            entries.append(init)
    return entries


def _walk(entry, scans) -> list[str]:
    steps, seen = [], set()

    def visit(fid):
        if fid in seen:
            return
        seen.add(fid)
        steps.append(fid)
        for t in scans[fid].targets:
            visit(t)
    visit(entry)
    return steps


# The page

def render(model: dict, out: Path) -> Path:
    page = (HERE / "page.html").read_text()
    library = (HERE / "cytoscape.min.js").read_text()
    data = json.dumps(model).replace("<", "\\u003c")  # no </script> or <!-- can end the data block early
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("__MODEL__", data, 1).replace("/*__CYTOSCAPE__*/", library, 1))
    return out


# Setup

def _git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _task() -> dict:
    script = "${userHome}/.claude/skills/code-map/scripts/code_map.py"
    return {"label": TASK_LABEL, "type": "shell", "command": f'python3 "{script}" "${{workspaceFolder}}" --open',
            "problemMatcher": [], "presentation": {"reveal": "silent"}}


def setup(root, ml: bool) -> dict:
    """Remember the ML mode, add the "Code map" VS Code task, and ignore both through .git/info/exclude, so the
    repo's tracked files and .gitignore never change."""
    root = Path(root).resolve()
    (root / OUT_DIR).mkdir(exist_ok=True)
    (root / OUT_DIR / "config.json").write_text(json.dumps({"ml": bool(ml)}) + "\n")
    ignore = [f"{OUT_DIR}/"]
    tasks_path = root / ".vscode" / "tasks.json"
    if _git(root, "ls-files", "--error-unmatch", ".vscode/tasks.json").returncode == 0:
        task = "skipped: .vscode/tasks.json is tracked by git"
    elif tasks_path.exists():
        tasks = json.loads(tasks_path.read_text())
        others = [t for t in tasks.get("tasks", []) if t.get("label") != TASK_LABEL]
        tasks["tasks"] = others + [_task()]
        tasks_path.write_text(json.dumps(tasks, indent=2) + "\n")
        task = "merged into .vscode/tasks.json"
        if not others:  # only ours, so an earlier setup created the file
            ignore.append(".vscode/tasks.json")
    else:
        tasks_path.parent.mkdir(exist_ok=True)
        tasks_path.write_text(json.dumps({"version": "2.0.0", "tasks": [_task()]}, indent=2) + "\n")
        task = "created .vscode/tasks.json"
        ignore.append(".vscode/tasks.json")
    git_dir = _git(root, "rev-parse", "--git-dir")
    excluded = []
    if git_dir.returncode == 0:
        exclude = (root / git_dir.stdout.strip() / "info" / "exclude").resolve()
        exclude.parent.mkdir(parents=True, exist_ok=True)
        lines = exclude.read_text().splitlines() if exclude.exists() else []
        excluded = [i for i in ignore if i not in lines]
        if excluded:
            exclude.write_text("\n".join([*lines, *excluded]) + "\n")
    return {"config": {"ml": bool(ml)}, "task": task, "excluded": excluded}


# Command line

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("detect", "setup"):
        p = argparse.ArgumentParser(prog=f"code_map.py {argv[0]}")
        p.add_argument("repo")
        if argv[0] == "setup":
            p.add_argument("--ml", action=argparse.BooleanOptionalAction, required=True)
        a = p.parse_args(argv[1:])
        try:
            result = detect(a.repo) if argv[0] == "detect" else setup(a.repo, a.ml)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0
    p = argparse.ArgumentParser(prog="code_map.py", description=__doc__.split("\n\n")[0])
    p.add_argument("repo")
    p.add_argument("--ml", action=argparse.BooleanOptionalAction, default=None,
                   help="show store and registry names (default: the setup answer, else detected)")
    p.add_argument("--out", type=Path, help=f"page path (default: <repo>/{OUT_DIR}/index.html)")
    p.add_argument("--json", action="store_true", help="also write model.json next to the page")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = p.parse_args(argv)
    root = Path(a.repo).resolve()
    try:
        model = build_model(root, a.ml)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = render(model, (a.out or root / OUT_DIR / "index.html").resolve())
    if a.json:
        (out.parent / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(out)
    for w in model["warnings"]:
        print(f"warning: {w}")
    if a.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
