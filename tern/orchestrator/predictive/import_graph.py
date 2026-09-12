from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from tern.orchestrator.project_intelligence_v2 import (
    ImportRecord,
    IndexedFile,
    ProjectSnapshotV2,
)


class ImportSourceKind(str, Enum):
    PRODUCTION = "PRODUCTION"
    TEST = "TEST"
    CONFIG = "CONFIG"
    SCRIPT = "SCRIPT"


@dataclass(frozen=True)
class ImportEdge:
    source: str
    target: str
    line: int
    scope: str
    source_kind: ImportSourceKind
    type_checking_only: bool = False
    symbol: str | None = None

    @property
    def runtime_production(self) -> bool:
        return (
            self.source_kind is ImportSourceKind.PRODUCTION
            and self.scope == "module"
            and not self.type_checking_only
        )


@dataclass(frozen=True)
class ImportCycleCause:
    strongly_connected_component: tuple[str, ...]
    production_edges: tuple[ImportEdge, ...]
    observer_edges: tuple[ImportEdge, ...]


def _kind(indexed: IndexedFile) -> ImportSourceKind:
    if indexed.is_test:
        return ImportSourceKind.TEST
    lowered = indexed.path.casefold()
    if "config" in lowered or "settings" in lowered:
        return ImportSourceKind.CONFIG
    if lowered.startswith(("scripts/", "bin/")):
        return ImportSourceKind.SCRIPT
    return ImportSourceKind.PRODUCTION


def _absolute_import(record: IndexedFile, imported: ImportRecord) -> str | None:
    if not imported.level:
        return imported.module
    package_parts = (record.package or "").split(".") if record.package else []
    remove = max(0, imported.level - 1)
    if remove > len(package_parts):
        return None
    prefix = package_parts[: len(package_parts) - remove]
    if imported.module:
        prefix.extend(imported.module.split("."))
    return ".".join(part for part in prefix if part) or None


def import_edges(snapshot: ProjectSnapshotV2) -> tuple[ImportEdge, ...]:
    modules = {item.module: item.path for item in snapshot.files if item.module}
    result: set[ImportEdge] = set()
    for indexed in snapshot.files:
        for imported in indexed.imports:
            base = _absolute_import(indexed, imported)
            names = ([base] if base else []) + [
                f"{base}.{name}" for name in imported.names if base
            ]
            if not base and imported.level and indexed.package:
                names.extend(f"{indexed.package}.{name}" for name in imported.names)
            for module in names:
                target = modules.get(module)
                if target and target != indexed.path:
                    symbol = next(
                        (
                            name for name in imported.names
                            if module.endswith(f".{name}")
                            or module == name
                        ),
                        imported.names[0] if len(imported.names) == 1 else None,
                    )
                    result.add(ImportEdge(
                        indexed.path, target, imported.line, imported.scope,
                        _kind(indexed), imported.type_checking_only, symbol,
                    ))
    return tuple(sorted(
        result,
        key=lambda item: (item.source, item.target, item.line, item.scope, item.symbol or ""),
    ))


def _tarjan(vertices: Iterable[str], edges: Iterable[ImportEdge]) -> tuple[tuple[str, ...], ...]:
    graph = {vertex: [] for vertex in vertices}
    for edge in edges:
        graph.setdefault(edge.source, []).append(edge.target)
    index = 0
    indexes: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(vertex: str) -> None:
        nonlocal index
        indexes[vertex] = low[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for neighbour in sorted(graph.get(vertex, ())):
            if neighbour not in indexes:
                visit(neighbour)
                low[vertex] = min(low[vertex], low[neighbour])
            elif neighbour in on_stack:
                low[vertex] = min(low[vertex], indexes[neighbour])
        if low[vertex] == indexes[vertex]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == vertex:
                    break
            if len(component) > 1:
                components.append(tuple(sorted(component)))

    for vertex in sorted(graph):
        if vertex not in indexes:
            visit(vertex)
    return tuple(sorted(components))


def find_import_cycles(snapshot: ProjectSnapshotV2) -> tuple[ImportCycleCause, ...]:
    edges = import_edges(snapshot)
    production = tuple(edge for edge in edges if edge.runtime_production)
    components = _tarjan(
        (item.path for item in snapshot.files if not item.is_test), production
    )
    causes: list[ImportCycleCause] = []
    for component in components:
        members = set(component)
        cycle_edges = tuple(
            edge for edge in production
            if edge.source in members and edge.target in members
        )
        observers = tuple(
            edge for edge in edges
            if edge.source not in members and edge.target in members
        )
        causes.append(ImportCycleCause(component, cycle_edges, observers))
    return tuple(causes)
