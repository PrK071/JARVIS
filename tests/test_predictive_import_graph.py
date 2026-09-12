from __future__ import annotations

from tern.orchestrator.predictive.import_graph import find_import_cycles, import_edges
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2


def _project(tmp_path, files):
    for path, source in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return ProjectIndexBuilderV2(tmp_path).build()


def test_tarjan_finds_three_node_production_cycle(tmp_path):
    snapshot = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "from pkg import b\n",
        "pkg/b.py": "from pkg import c\n",
        "pkg/c.py": "from pkg import a\n",
    })

    cycles = find_import_cycles(snapshot)

    assert len(cycles) == 1
    assert cycles[0].strongly_connected_component == (
        "pkg/a.py", "pkg/b.py", "pkg/c.py"
    )
    assert len(cycles[0].production_edges) == 3


def test_test_import_is_observer_not_cycle_member(tmp_path):
    snapshot = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "from pkg import b\n",
        "pkg/b.py": "from pkg import a\n",
        "tests/test_a.py": "from pkg import a\n",
    })

    cycle = find_import_cycles(snapshot)[0]

    assert cycle.strongly_connected_component == ("pkg/a.py", "pkg/b.py")
    assert [edge.source for edge in cycle.observer_edges] == ["tests/test_a.py"]


def test_type_checking_and_local_imports_are_not_runtime_cycle_edges(tmp_path):
    snapshot = _project(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from pkg import b\n",
        "pkg/b.py": "def load():\n    from pkg import a\n",
    })

    edges = import_edges(snapshot)

    assert any(edge.type_checking_only for edge in edges)
    assert any(edge.scope == "load" for edge in edges)
    assert find_import_cycles(snapshot) == ()
