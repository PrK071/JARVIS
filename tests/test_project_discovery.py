from __future__ import annotations

from pathlib import Path

from tern.orchestrator.project_discovery import DiscoveryPolicy, ProjectDiscovery


def _project(root: Path, name: str, *, marker: str = "package.json") -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / marker).write_text('{"name":"%s"}' % name, encoding="utf-8")
    return path


def test_resolves_exact_project_and_returns_read_only_evidence(tmp_path: Path) -> None:
    projects = tmp_path / "Projects"
    projects.mkdir()
    target = _project(projects, "Kari")
    result = ProjectDiscovery(
        DiscoveryPolicy.from_values([projects], max_depth=2)
    ).discover("Kari")

    assert result.status == "RESOLVED"
    assert result.candidates[0].path == str(target.resolve())
    assert "PROJECT_MARKER_FOUND" in result.candidates[0].match_reasons


def test_ambiguous_exact_candidates_are_not_selected(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    _project(first, "Kari")
    _project(second, "Kari")

    result = ProjectDiscovery(
        DiscoveryPolicy.from_values([tmp_path], max_depth=3)
    ).discover("Kari")

    assert result.status == "AMBIGUOUS"
    assert len(result.candidates) == 2


def test_excluded_directories_are_not_traversed(tmp_path: Path) -> None:
    excluded = tmp_path / "node_modules"
    excluded.mkdir()
    _project(excluded, "Kari")

    result = ProjectDiscovery(
        DiscoveryPolicy.from_values([tmp_path], max_depth=3)
    ).discover("Kari")

    assert result.status == "NOT_FOUND"
    assert not result.candidates


def test_discovery_budget_terminates(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    result = ProjectDiscovery(
        DiscoveryPolicy.from_values([tmp_path], max_depth=5, max_directories=1)
    ).discover("Kari")

    assert result.budget_reached
    assert result.directories_checked == 1
