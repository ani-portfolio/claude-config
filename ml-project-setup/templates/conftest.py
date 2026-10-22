"""Mark each test by the level folder it lives in, in tests/ or in a pipeline's tests/, so `pytest -m unit`
(or component, system) selects one level across the whole project."""

import pytest

LEVELS = ("unit", "component", "system")


def pytest_collection_modifyitems(items):
    for item in items:
        level = next((part for part in item.path.parts if part in LEVELS), None)
        if level:
            item.add_marker(getattr(pytest.mark, level))
