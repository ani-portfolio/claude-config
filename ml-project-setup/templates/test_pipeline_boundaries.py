"""The pipelines talk only through the feature store and the model registry: none imports another pipeline's code,
and the shared modules in __SRC__/ import no pipeline. Installed by ml-project-setup (--add-deployment)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "__SRC__"
PIPELINES = __PIPELINES__


def imported_modules(path):
    """Every module a file imports, as an absolute dotted name starting with the src package. Relative imports are
    resolved against the file's package."""
    package = [SRC.name, *path.relative_to(SRC).parts[:-1]]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            yield module
            yield from (f"{module}.{a.name}" for a in node.names)


def imported_pipelines(path):
    for name in imported_modules(path):
        parts = name.split(".")
        if len(parts) > 1 and parts[0] == SRC.name and parts[1] in PIPELINES:
            yield parts[1]


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_pipeline_imports_no_other_pipeline(pipeline):
    bad = {f"{p.relative_to(SRC)} imports {name}" for p in (SRC / pipeline).rglob("*.py")
           for name in imported_pipelines(p) if name != pipeline}
    assert not bad, sorted(bad)


def test_shared_code_imports_no_pipeline():
    shared = [p for p in SRC.rglob("*.py") if p.relative_to(SRC).parts[0] not in PIPELINES]
    bad = {f"{p.relative_to(SRC)} imports {name}" for p in shared for name in imported_pipelines(p)}
    assert not bad, sorted(bad)
