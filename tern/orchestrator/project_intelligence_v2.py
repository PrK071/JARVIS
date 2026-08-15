from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .project_intelligence import (
    CONFIG_FILES,
    DEPENDENCY_FILES,
    IGNORED_DIRECTORIES,
    LANGUAGES,
)
from .security import PathPolicy


PYTHON_SUFFIX = ".py"
INDEX_VERSION = 2


class SymbolKind(str, Enum):
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CONSTANT = "constant"
    CLASS_ATTRIBUTE = "class_attribute"


class RelevantFileEvidenceSource(str, Enum):
    EXPLICIT_FILE_REFERENCE = "EXPLICIT_FILE_REFERENCE"
    EXPLICIT_DIRECTORY_REFERENCE = "EXPLICIT_DIRECTORY_REFERENCE"
    EXPLICIT_SYMBOL_REFERENCE = "EXPLICIT_SYMBOL_REFERENCE"
    SYMBOL_DEFINITION = "SYMBOL_DEFINITION"
    SYMBOL_REFERENCE = "SYMBOL_REFERENCE"
    IMPORT_DEPENDENCY = "IMPORT_DEPENDENCY"
    REVERSE_IMPORT = "REVERSE_IMPORT"
    TEST_RELATIONSHIP = "TEST_RELATIONSHIP"
    TRACEBACK_REFERENCE = "TRACEBACK_REFERENCE"
    ERROR_REFERENCE = "ERROR_REFERENCE"
    GIT_MODIFIED_FILE = "GIT_MODIFIED_FILE"
    GIT_DIFF_RELATIONSHIP = "GIT_DIFF_RELATIONSHIP"
    CONFIG_RELATIONSHIP = "CONFIG_RELATIONSHIP"
    ENTRYPOINT_RELATIONSHIP = "ENTRYPOINT_RELATIONSHIP"
    PROJECT_STRUCTURE = "PROJECT_STRUCTURE"
    SEMANTIC_INFERENCE = "SEMANTIC_INFERENCE"


class EvidenceStrength(str, Enum):
    HARD = "HARD"
    STRONG = "STRONG"
    SUPPORTING = "SUPPORTING"
    SEMANTIC = "SEMANTIC"


_STRENGTH_ORDER = {
    EvidenceStrength.HARD: 4,
    EvidenceStrength.STRONG: 3,
    EvidenceStrength.SUPPORTING: 2,
    EvidenceStrength.SEMANTIC: 1,
}


_SOURCE_ORDER = {
    RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE: 16,
    RelevantFileEvidenceSource.TRACEBACK_REFERENCE: 15,
    RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE: 14,
    RelevantFileEvidenceSource.SYMBOL_DEFINITION: 13,
    RelevantFileEvidenceSource.ERROR_REFERENCE: 12,
    RelevantFileEvidenceSource.TEST_RELATIONSHIP: 11,
    RelevantFileEvidenceSource.IMPORT_DEPENDENCY: 10,
    RelevantFileEvidenceSource.REVERSE_IMPORT: 9,
    RelevantFileEvidenceSource.SYMBOL_REFERENCE: 8,
    RelevantFileEvidenceSource.EXPLICIT_DIRECTORY_REFERENCE: 7,
    RelevantFileEvidenceSource.GIT_DIFF_RELATIONSHIP: 6,
    RelevantFileEvidenceSource.CONFIG_RELATIONSHIP: 5,
    RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP: 4,
    RelevantFileEvidenceSource.GIT_MODIFIED_FILE: 3,
    RelevantFileEvidenceSource.PROJECT_STRUCTURE: 2,
    RelevantFileEvidenceSource.SEMANTIC_INFERENCE: 1,
}


@dataclass(frozen=True)
class ProjectIndexBudget:
    max_files: int = 10_000
    max_python_file_bytes: int = 1_048_576
    max_total_python_bytes: int = 64 * 1_048_576


@dataclass(frozen=True)
class CandidateBudget:
    max_candidates: int = 32
    max_selected_files: int = 12
    max_context_bytes: int = 48_000
    max_context_bytes_per_file: int = 12_000
    max_symbol_lines: int = 240


@dataclass(frozen=True)
class SymbolRecord:
    name: str
    qualified_name: str
    kind: SymbolKind
    file: str
    line: int
    end_line: int
    module: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SymbolRecord":
        return cls(
            str(value["name"]),
            str(value["qualified_name"]),
            SymbolKind(value["kind"]),
            str(value["file"]),
            int(value["line"]),
            int(value["end_line"]),
            str(value["module"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "kind": self.kind.value,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "module": self.module,
        }


@dataclass(frozen=True)
class ImportRecord:
    module: str | None
    names: tuple[str, ...]
    level: int
    line: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImportRecord":
        return cls(
            str(value["module"]) if value.get("module") else None,
            tuple(str(item) for item in value.get("names") or ()),
            int(value.get("level") or 0),
            int(value.get("line") or 0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": list(self.names),
            "level": self.level,
            "line": self.line,
        }


@dataclass(frozen=True)
class IndexedFile:
    path: str
    language: str
    size: int
    modified_ns: int
    sha256: str
    module: str | None
    package: str | None
    symbols: tuple[SymbolRecord, ...]
    imports: tuple[ImportRecord, ...]
    referenced_names: tuple[str, ...]
    is_test: bool
    is_config: bool
    is_dependency: bool
    is_entrypoint: bool
    analyzed: bool
    parse_error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndexedFile":
        return cls(
            str(value["path"]),
            str(value["language"]),
            int(value["size"]),
            int(value["modified_ns"]),
            str(value["sha256"]),
            str(value["module"]) if value.get("module") else None,
            str(value["package"]) if value.get("package") else None,
            tuple(SymbolRecord.from_dict(item) for item in value.get("symbols") or ()),
            tuple(ImportRecord.from_dict(item) for item in value.get("imports") or ()),
            tuple(str(item) for item in value.get("referenced_names") or ()),
            bool(value.get("is_test")),
            bool(value.get("is_config")),
            bool(value.get("is_dependency")),
            bool(value.get("is_entrypoint")),
            bool(value.get("analyzed")),
            str(value["parse_error"]) if value.get("parse_error") else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "size": self.size,
            "modified_ns": self.modified_ns,
            "sha256": self.sha256,
            "module": self.module,
            "package": self.package,
            "symbols": [item.as_dict() for item in self.symbols],
            "imports": [item.as_dict() for item in self.imports],
            "referenced_names": list(self.referenced_names),
            "is_test": self.is_test,
            "is_config": self.is_config,
            "is_dependency": self.is_dependency,
            "is_entrypoint": self.is_entrypoint,
            "analyzed": self.analyzed,
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class TestRelationship:
    production_file: str
    test_file: str
    evidence: tuple[str, ...]
    strength: EvidenceStrength

    def as_dict(self) -> dict[str, Any]:
        return {
            "production_file": self.production_file,
            "test_file": self.test_file,
            "evidence": list(self.evidence),
            "strength": self.strength.value,
        }


@dataclass(frozen=True)
class ProjectIndexMetrics:
    repo_scan_ms: float
    index_build_ms: float
    files_discovered: int
    files_reused: int
    files_indexed: int
    bytes_hashed: int
    bytes_parsed: int
    cache_hit: bool

    @property
    def incremental_update_ms(self) -> float:
        return self.index_build_ms if self.cache_hit else 0.0


@dataclass(frozen=True)
class ProjectSnapshotV2:
    project_path: str
    project_id: str
    languages: tuple[str, ...]
    directories: tuple[str, ...]
    files: tuple[IndexedFile, ...]
    entry_points: tuple[str, ...]
    declared_entry_points: tuple[str, ...]
    config_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    test_roots: tuple[str, ...]
    git_branch: str | None
    modified_files: tuple[str, ...]
    untracked_files: tuple[str, ...]
    diff_files: tuple[str, ...]
    import_graph: Mapping[str, tuple[str, ...]]
    reverse_import_graph: Mapping[str, tuple[str, ...]]
    test_relationships: tuple[TestRelationship, ...]
    known_errors: tuple[str, ...]
    known_traceback_files: tuple[str, ...]
    metrics: ProjectIndexMetrics

    @property
    def file_index(self) -> dict[str, IndexedFile]:
        return {item.path: item for item in self.files}

    @property
    def symbol_index(self) -> dict[str, tuple[SymbolRecord, ...]]:
        values: dict[str, list[SymbolRecord]] = {}
        for item in self.files:
            for symbol in item.symbols:
                for key in {symbol.name, symbol.qualified_name}:
                    values.setdefault(key, []).append(symbol)
        return {
            key: tuple(sorted(records, key=lambda item: (item.file, item.line)))
            for key, records in values.items()
        }

    def compact(self, *, max_files: int = 80, max_bytes: int = 24_000) -> dict[str, Any]:
        ordered = sorted(
            self.files,
            key=lambda item: (
                item.path not in self.modified_files,
                not item.is_entrypoint,
                not item.is_config,
                item.path.casefold(),
            ),
        )
        payload: dict[str, Any] = {
            "project_path": self.project_path,
            "project_id": self.project_id,
            "languages": list(self.languages),
            "entry_points": list(self.entry_points),
            "declared_entry_points": list(self.declared_entry_points),
            "config_files": list(self.config_files),
            "dependency_files": list(self.dependency_files),
            "test_roots": list(self.test_roots),
            "git_branch": self.git_branch,
            "modified_files": list(self.modified_files),
            "untracked_files": list(self.untracked_files),
            "files": [],
            "file_total": len(self.files),
            "truncated": len(self.files) > max_files,
        }
        for item in ordered[:max_files]:
            payload["files"].append(
                {
                    "path": item.path,
                    "module": item.module,
                    "symbols": [symbol.qualified_name for symbol in item.symbols[:8]],
                    "imports": list(self.import_graph.get(item.path, ()))[:8],
                    "is_test": item.is_test,
                    "is_config": item.is_config,
                    "is_entrypoint": item.is_entrypoint,
                    "size": item.size,
                }
            )
            if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_bytes:
                payload["files"].pop()
                payload["truncated"] = True
                break
        return payload


def _module_for_path(path: str) -> tuple[str | None, str | None]:
    if not path.endswith(PYTHON_SUFFIX):
        return None, None
    raw = path[:-3].replace("/", ".")
    is_package = raw.endswith(".__init__")
    module = raw.removesuffix(".__init__")
    package = module if is_package else module.rpartition(".")[0] or None
    return module, package


def _symbol_records(tree: ast.AST, *, path: str, module: str) -> tuple[SymbolRecord, ...]:
    records: list[SymbolRecord] = []

    def visit(nodes: Sequence[ast.stmt], owner: str | None = None) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                qualified = f"{owner}.{node.name}" if owner else node.name
                records.append(
                    SymbolRecord(
                        node.name,
                        qualified,
                        SymbolKind.CLASS,
                        path,
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        module,
                    )
                )
                visit(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{owner}.{node.name}" if owner else node.name
                if owner:
                    kind = (
                        SymbolKind.ASYNC_METHOD
                        if isinstance(node, ast.AsyncFunctionDef)
                        else SymbolKind.METHOD
                    )
                else:
                    kind = (
                        SymbolKind.ASYNC_FUNCTION
                        if isinstance(node, ast.AsyncFunctionDef)
                        else SymbolKind.FUNCTION
                    )
                records.append(
                    SymbolRecord(
                        node.name,
                        qualified,
                        kind,
                        path,
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        module,
                    )
                )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if not (target.id.isupper() or owner):
                        continue
                    qualified = f"{owner}.{target.id}" if owner else target.id
                    records.append(
                        SymbolRecord(
                            target.id,
                            qualified,
                            SymbolKind.CLASS_ATTRIBUTE if owner else SymbolKind.CONSTANT,
                            path,
                            node.lineno,
                            getattr(node, "end_lineno", node.lineno),
                            module,
                        )
                    )

    visit(getattr(tree, "body", ()))
    return tuple(records)


def _import_records(tree: ast.AST) -> tuple[ImportRecord, ...]:
    records: list[ImportRecord] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(ImportRecord(alias.name, (), 0, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            records.append(
                ImportRecord(
                    node.module,
                    tuple(alias.name for alias in node.names),
                    int(node.level or 0),
                    node.lineno,
                )
            )
    return tuple(records)


def _referenced_names(tree: ast.AST) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return tuple(sorted(names))


class ProjectIndexBuilderV2:
    def __init__(
        self,
        project_path: str | Path,
        *,
        cache_path: str | Path | None = None,
        path_policy: PathPolicy | None = None,
        budget: ProjectIndexBudget | None = None,
    ) -> None:
        if path_policy:
            self.root = path_policy.resolve(str(project_path))
        else:
            self.root = Path(project_path).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("project_path must be a directory")
        self.path_policy = path_policy
        self.cache_path = Path(cache_path) if cache_path else None
        self.budget = budget or ProjectIndexBudget()
        self.project_id = hashlib.sha256(
            os.path.normcase(str(self.root)).encode("utf-8")
        ).hexdigest()[:20]

    def build(
        self,
        *,
        known_errors: Iterable[str] = (),
        known_traceback_files: Iterable[str] = (),
    ) -> ProjectSnapshotV2:
        started = time.perf_counter()
        paths = self._project_files()
        scan_ms = (time.perf_counter() - started) * 1000
        previous = self._load_cache()
        records: list[IndexedFile] = []
        reused = indexed = bytes_hashed = bytes_parsed = 0
        python_bytes = 0
        for path in paths:
            relative = path.relative_to(self.root).as_posix()
            if self.path_policy:
                self.path_policy.resolve(str(path))
            raw = path.read_bytes()
            bytes_hashed += len(raw)
            digest = hashlib.sha256(raw).hexdigest()
            stat = path.stat()
            old = previous.get(relative)
            if old and old.sha256 == digest:
                records.append(
                    replace(old, modified_ns=stat.st_mtime_ns, size=stat.st_size)
                )
                reused += 1
                continue
            can_parse = bool(
                path.suffix.casefold() == PYTHON_SUFFIX
                and len(raw) <= self.budget.max_python_file_bytes
                and python_bytes + len(raw) <= self.budget.max_total_python_bytes
            )
            record = self._index_file(
                relative,
                raw,
                size=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                digest=digest,
                parse_python=can_parse,
            )
            records.append(record)
            indexed += 1
            if can_parse:
                python_bytes += len(raw)
                bytes_parsed += len(raw)

        records.sort(key=lambda item: item.path.casefold())
        entry_points, declared_entry_points = self._entry_points(records)
        if entry_points:
            records = [
                replace(item, is_entrypoint=item.path in entry_points)
                if item.path in entry_points
                else item
                for item in records
            ]
        import_graph = self._resolve_import_graph(records)
        reverse_graph = self._reverse_graph(import_graph)
        test_relationships = self._test_relationships(records, import_graph)
        git = self._git_facts()
        self._save_cache(records)
        build_ms = (time.perf_counter() - started) * 1000
        directories = sorted(
            {
                str(PurePosixPath(item.path).parent)
                for item in records
                if str(PurePosixPath(item.path).parent) != "."
            }
        )
        test_roots = sorted(
            {
                item.path.split("/", 1)[0]
                for item in records
                if item.is_test and "/" in item.path
            }
        )
        metrics = ProjectIndexMetrics(
            round(scan_ms, 3),
            round(build_ms, 3),
            len(paths),
            reused,
            indexed,
            bytes_hashed,
            bytes_parsed,
            bool(previous),
        )
        return ProjectSnapshotV2(
            str(self.root),
            self.project_id,
            tuple(sorted({item.language for item in records})),
            tuple(directories),
            tuple(records),
            tuple(sorted(entry_points)),
            tuple(sorted(declared_entry_points)),
            tuple(sorted(item.path for item in records if item.is_config)),
            tuple(sorted(item.path for item in records if item.is_dependency)),
            tuple(test_roots),
            git["branch"],
            tuple(git["modified"]),
            tuple(git["untracked"]),
            tuple(git["diff"]),
            import_graph,
            reverse_graph,
            test_relationships,
            tuple(known_errors),
            tuple(known_traceback_files),
            metrics,
        )

    def _project_files(self) -> list[Path]:
        started: list[Path] = []
        if (self.root / ".git").exists():
            completed = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                for raw_name in completed.stdout.decode("utf-8", errors="replace").split("\0"):
                    if not raw_name:
                        continue
                    candidate = self.root / raw_name
                    if candidate.is_symlink():
                        continue
                    path = candidate.resolve()
                    try:
                        path.relative_to(self.root)
                    except ValueError:
                        continue
                    if path.is_file():
                        started.append(path)
                    if len(started) >= self.budget.max_files:
                        break
                return sorted(set(started), key=lambda item: str(item).casefold())
        for current, directories, files in os.walk(self.root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name.casefold() not in IGNORED_DIRECTORIES
                and not Path(current, name).is_symlink()
            ]
            for name in files:
                path = Path(current, name)
                if path.is_symlink():
                    continue
                started.append(path.resolve())
                if len(started) >= self.budget.max_files:
                    return started
        return started

    def _index_file(
        self,
        relative: str,
        raw: bytes,
        *,
        size: int,
        modified_ns: int,
        digest: str,
        parse_python: bool,
    ) -> IndexedFile:
        suffix = Path(relative).suffix.casefold()
        module, package = _module_for_path(relative)
        symbols: tuple[SymbolRecord, ...] = ()
        imports: tuple[ImportRecord, ...] = ()
        references: tuple[str, ...] = ()
        parse_error = None
        if parse_python and module:
            try:
                tree = ast.parse(raw.decode("utf-8"), filename=relative)
                symbols = _symbol_records(tree, path=relative, module=module)
                imports = _import_records(tree)
                references = _referenced_names(tree)
            except (SyntaxError, UnicodeDecodeError) as exc:
                parse_error = type(exc).__name__
        basename = Path(relative).name.casefold()
        is_test = bool(
            relative.startswith(("tests/", "test/"))
            or basename.startswith("test_")
            or basename.endswith("_test.py")
        )
        return IndexedFile(
            relative,
            LANGUAGES.get(suffix, suffix.lstrip(".") or "file"),
            size,
            modified_ns,
            digest,
            module,
            package,
            symbols,
            imports,
            references,
            is_test,
            basename in CONFIG_FILES,
            basename in DEPENDENCY_FILES,
            basename in {"main.py", "__main__.py"},
            parse_python,
            parse_error,
        )

    def _entry_points(
        self, records: Sequence[IndexedFile]
    ) -> tuple[set[str], set[str]]:
        result = {
            item.path
            for item in records
            if Path(item.path).name in {"main.py", "__main__.py"}
        }
        declared: set[str] = set()
        by_module = {item.module: item.path for item in records if item.module}
        pyproject = self.root / "pyproject.toml"
        if pyproject.is_file():
            try:
                value = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                scripts = value.get("project", {}).get("scripts", {})
                for target in scripts.values():
                    module = str(target).split(":", 1)[0]
                    if module in by_module:
                        result.add(by_module[module])
                        declared.add(by_module[module])
            except (OSError, tomllib.TOMLDecodeError):
                pass
        return result, declared

    @staticmethod
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

    def _resolve_import_graph(
        self, records: Sequence[IndexedFile]
    ) -> dict[str, tuple[str, ...]]:
        modules = {item.module: item.path for item in records if item.module}
        graph: dict[str, tuple[str, ...]] = {}
        for item in records:
            resolved: set[str] = set()
            for imported in item.imports:
                base = self._absolute_import(item, imported)
                candidates = [base] if base else []
                candidates.extend(
                    f"{base}.{name}" for name in imported.names if base
                )
                if not base and imported.level and item.package:
                    candidates.extend(
                        f"{item.package}.{name}" for name in imported.names
                    )
                for candidate in candidates:
                    if candidate in modules and modules[candidate] != item.path:
                        resolved.add(modules[candidate])
            graph[item.path] = tuple(sorted(resolved))
        return graph

    @staticmethod
    def _reverse_graph(
        graph: Mapping[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {path: [] for path in graph}
        for source, targets in graph.items():
            for target in targets:
                result.setdefault(target, []).append(source)
        return {
            path: tuple(sorted(set(sources)))
            for path, sources in result.items()
        }

    @staticmethod
    def _test_relationships(
        records: Sequence[IndexedFile],
        graph: Mapping[str, tuple[str, ...]],
    ) -> tuple[TestRelationship, ...]:
        production = [item for item in records if not item.is_test and item.module]
        tests = [item for item in records if item.is_test]
        result: dict[tuple[str, str], TestRelationship] = {}
        for test in tests:
            imports = set(graph.get(test.path, ()))
            for target in production:
                evidence: list[str] = []
                strength = EvidenceStrength.SUPPORTING
                if target.path in imports:
                    evidence.append("direct_import")
                    strength = EvidenceStrength.STRONG
                target_symbols = {symbol.name for symbol in target.symbols}
                if target.path in imports and target_symbols & set(test.referenced_names):
                    evidence.append("symbol_reference")
                production_stem = Path(target.path).stem.removeprefix("__init__")
                test_stem = Path(test.path).stem.removeprefix("test_").removesuffix("_test")
                if production_stem and production_stem == test_stem:
                    evidence.append("naming_convention")
                if evidence:
                    result[(target.path, test.path)] = TestRelationship(
                        target.path,
                        test.path,
                        tuple(evidence),
                        strength,
                    )
        return tuple(sorted(result.values(), key=lambda item: (item.production_file, item.test_file)))

    def _git_facts(self) -> dict[str, Any]:
        branch = subprocess.run(
            ["git", "-C", str(self.root), "branch", "--show-current"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        ).stdout.strip() or None
        completed = subprocess.run(
            ["git", "-C", str(self.root), "status", "--porcelain=v1", "-z"],
            capture_output=True,
            check=False,
        )
        modified: list[str] = []
        untracked: list[str] = []
        if completed.returncode == 0:
            for record in completed.stdout.decode("utf-8", errors="replace").split("\0"):
                if not record:
                    continue
                code, raw_path = record[:2], record[3:]
                path = raw_path.split(" -> ")[-1].replace("\\", "/")
                (untracked if code == "??" else modified).append(path)
        diff = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--name-only", "HEAD", "--"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return {
            "branch": branch,
            "modified": sorted(set(modified)),
            "untracked": sorted(set(untracked)),
            "diff": sorted(set(line.replace("\\", "/") for line in diff.stdout.splitlines() if line)),
        }

    def _load_cache(self) -> dict[str, IndexedFile]:
        if self.cache_path is None:
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if (
                value.get("version") != INDEX_VERSION
                or value.get("project_id") != self.project_id
                or value.get("project_path") != str(self.root)
            ):
                return {}
            return {
                str(item["path"]): IndexedFile.from_dict(item)
                for item in value.get("files") or ()
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return {}

    def _save_cache(self, records: Sequence[IndexedFile]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "project_id": self.project_id,
            "project_path": str(self.root),
            "files": [item.as_dict() for item in records],
        }
        handle, temporary_name = tempfile.mkstemp(
            prefix="project-index-v2-",
            suffix=".json",
            dir=self.cache_path.parent,
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RelevantFileEvidence:
    source: RelevantFileEvidenceSource
    strength: EvidenceStrength
    target: str
    relationship: str
    evidence_ref: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "strength": self.strength.value,
            "target": self.target,
            "relationship": self.relationship,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class RelevantFileCandidate:
    path: str
    evidences: tuple[RelevantFileEvidence, ...]
    size: int
    language: str
    module: str | None
    is_test: bool

    @property
    def strongest(self) -> EvidenceStrength:
        return max(
            (item.strength for item in self.evidences),
            key=lambda item: _STRENGTH_ORDER[item],
        )

    @property
    def hard(self) -> bool:
        return self.strongest is EvidenceStrength.HARD

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "evidences": [item.as_dict() for item in self.evidences],
            "strongest": self.strongest.value,
            "size": self.size,
            "language": self.language,
            "module": self.module,
            "is_test": self.is_test,
        }


@dataclass(frozen=True)
class ContextSlice:
    path: str
    start_line: int
    end_line: int
    bytes_selected: int
    estimated_tokens: int
    evidence_ref: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "bytes_selected": self.bytes_selected,
            "estimated_tokens": self.estimated_tokens,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class CandidateGenerationMetrics:
    generation_ms: float
    semantic_ranking_ms: float
    candidate_count: int
    selected_count: int
    selected_file_bytes: int
    context_bytes: int
    estimated_context_tokens: int
    context_budget_violation: bool
    hard_evidence_dropped: int
    semantic_ranking_used: bool


@dataclass(frozen=True)
class ProjectCandidateSelection:
    project_path: str
    project_id: str
    candidates: tuple[RelevantFileCandidate, ...]
    selected: tuple[RelevantFileCandidate, ...]
    context_slices: tuple[ContextSlice, ...]
    read_scope: tuple[str, ...]
    allowed_mutation_targets: tuple[str, ...]
    forbidden_mutation_targets: tuple[str, ...]
    metrics: CandidateGenerationMetrics
    dry_run: bool = True
    execution_authorized: bool = False

    @property
    def selected_files(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "project_id": self.project_id,
            "candidates": [item.as_dict() for item in self.candidates],
            "selected": [item.as_dict() for item in self.selected],
            "context_slices": [item.as_dict() for item in self.context_slices],
            "read_scope": list(self.read_scope),
            "allowed_mutation_targets": list(self.allowed_mutation_targets),
            "forbidden_mutation_targets": list(self.forbidden_mutation_targets),
            "metrics": self.metrics.__dict__,
            "dry_run": self.dry_run,
            "execution_authorized": self.execution_authorized,
        }

    def future_worker_handoff(self, *, task: str, requirements: Mapping[str, Any]) -> dict[str, Any]:
        """Describe a future payload without delegating or reading more files."""
        return {
            "task": task,
            "requirements": dict(requirements),
            "relevant_files": [
                {
                    "path": item.path,
                    "evidence": [evidence.as_dict() for evidence in item.evidences],
                }
                for item in self.selected
            ],
            "constraints": {
                "allowed_mutation_targets": list(self.allowed_mutation_targets),
                "forbidden_mutation_targets": list(self.forbidden_mutation_targets),
            },
            "dry_run": True,
            "execution_authorized": False,
        }


class _EvidenceAggregator:
    def __init__(self, snapshot: ProjectSnapshotV2):
        self.snapshot = snapshot
        self.values: dict[str, list[RelevantFileEvidence]] = {}

    def add(
        self,
        path: str,
        source: RelevantFileEvidenceSource,
        strength: EvidenceStrength,
        target: str,
        relationship: str,
        evidence_ref: str,
    ) -> None:
        if path not in self.snapshot.file_index:
            return
        evidence = RelevantFileEvidence(
            source, strength, target, relationship, evidence_ref
        )
        values = self.values.setdefault(path, [])
        if evidence not in values:
            values.append(evidence)

    def candidate(self, path: str) -> RelevantFileCandidate:
        item = self.snapshot.file_index[path]
        evidence = tuple(
            sorted(
                self.values[path],
                key=lambda value: (
                    -_STRENGTH_ORDER[value.strength],
                    -_SOURCE_ORDER[value.source],
                    value.evidence_ref,
                ),
            )
        )
        return RelevantFileCandidate(
            path,
            evidence,
            item.size,
            item.language,
            item.module,
            item.is_test,
        )


def _candidate_sort_key(candidate: RelevantFileCandidate) -> tuple[Any, ...]:
    return (
        -_STRENGTH_ORDER[candidate.strongest],
        -max(_SOURCE_ORDER[item.source] for item in candidate.evidences),
        -len(candidate.evidences),
        candidate.path.casefold(),
    )


def project_candidate_ranking_schema(paths: Sequence[str], max_items: int) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_files": {
                "type": "array",
                "items": {"type": "string", "enum": list(paths)},
                "uniqueItems": True,
                "maxItems": max_items,
            }
        },
        "required": ["selected_files"],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "project_candidate_ranking",
            "strict": True,
            "schema": schema,
        },
    }


class ProjectCandidateRanker:
    system_prompt = (
        "Choose a small sufficient subset only from the supplied project candidates. "
        "Use their structural evidence and grounded task requirements. Do not remove "
        "hard evidence, invent paths, select an agent, propose edits, or execute tools."
    )

    def __init__(self, client: Any):
        self.client = client

    def rank(
        self,
        task: str,
        candidates: Sequence[RelevantFileCandidate],
        *,
        requirements: Mapping[str, Any],
        max_items: int,
    ) -> tuple[str, ...] | None:
        paths = tuple(item.path for item in candidates)
        if not paths or max_items <= 0:
            return ()
        compact = [
            {
                "path": item.path,
                "module": item.module,
                "is_test": item.is_test,
                "evidence": [
                    {
                        "source": evidence.source.value,
                        "strength": evidence.strength.value,
                        "relationship": evidence.relationship,
                        "target": evidence.target,
                    }
                    for evidence in item.evidences
                ],
            }
            for item in candidates
        ]
        try:
            response = self.client.chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": task,
                                "task_requirements": requirements,
                                "candidates": compact,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format=project_candidate_ranking_schema(paths, max_items),
                temperature=0.0,
                max_tokens=256,
            )
            value = json.loads(response["choices"][0]["message"]["content"])
            selected = tuple(str(item) for item in value["selected_files"])
            if not set(selected).issubset(paths):
                return None
            return selected
        except Exception:
            return None


class ProjectCandidateGenerator:
    _TRACEBACK = re.compile(
        r"File\s+[\"'](?P<path>[^\"']+)[\"']\s*,\s*line\s+(?P<line>\d+)",
        re.IGNORECASE,
    )
    _PATH_LINE = re.compile(
        r"(?P<path>(?:[A-Za-z]:)?[^\s:\"']+\.py):(?P<line>\d+)"
    )
    _IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_.]*)(?![A-Za-z0-9_])")
    _TASK_TOKEN = re.compile(r"[A-Za-z0-9_]{4,}")

    def __init__(
        self,
        *,
        budget: CandidateBudget | None = None,
        ranker: ProjectCandidateRanker | None = None,
    ) -> None:
        self.budget = budget or CandidateBudget()
        self.ranker = ranker

    def generate(
        self,
        task: str,
        snapshot: ProjectSnapshotV2,
        *,
        requirements: Any | None = None,
    ) -> ProjectCandidateSelection:
        started = time.perf_counter()
        aggregator = _EvidenceAggregator(snapshot)
        task_normalized = task.replace("\\", "/")
        task_casefold = task_normalized.casefold()
        for path in snapshot.known_traceback_files:
            normalized = path.replace("\\", "/").removeprefix("./")
            aggregator.add(
                normalized,
                RelevantFileEvidenceSource.TRACEBACK_REFERENCE,
                EvidenceStrength.HARD,
                normalized,
                "known_traceback_file",
                "runtime:known_traceback_file",
            )
        self._explicit_files(task_casefold, snapshot, aggregator)
        self._tracebacks(task_normalized, snapshot, aggregator)
        self._symbols(task_normalized, snapshot, aggregator)
        self._structure(task_casefold, snapshot, aggregator)
        strong_seeds = tuple(
            path
            for path, evidences in aggregator.values.items()
            if max(_STRENGTH_ORDER[item.strength] for item in evidences)
            >= _STRENGTH_ORDER[EvidenceStrength.STRONG]
        )
        structural_seeds = tuple(
            path
            for path, evidences in aggregator.values.items()
            if any(
                item.source is RelevantFileEvidenceSource.PROJECT_STRUCTURE
                for item in evidences
            )
            and snapshot.file_index[path].language == "Python"
            and not snapshot.file_index[path].is_test
        )
        seeds = strong_seeds
        if not seeds and len(structural_seeds) <= 3:
            seeds = structural_seeds
        self._relationships(seeds, snapshot, aggregator, requirements=requirements)
        self._git_evidence(snapshot, aggregator)

        candidates = tuple(
            sorted(
                (aggregator.candidate(path) for path in aggregator.values),
                key=_candidate_sort_key,
            )
        )
        hard = tuple(item for item in candidates if item.hard)
        soft = tuple(item for item in candidates if not item.hard)
        candidate_limit = max(self.budget.max_candidates, len(hard))
        bounded_soft = soft[: max(0, candidate_limit - len(hard))]
        candidates = (*hard, *bounded_soft)
        selected_limit = max(self.budget.max_selected_files, len(hard))
        remaining = max(0, selected_limit - len(hard))
        semantic_ms = 0.0
        semantic_used = False
        selected_soft = bounded_soft[:remaining]
        if self.ranker and len(bounded_soft) > remaining and remaining:
            semantic_started = time.perf_counter()
            ranked = self.ranker.rank(
                task,
                bounded_soft,
                requirements=self._requirements_payload(requirements),
                max_items=remaining,
            )
            semantic_ms = (time.perf_counter() - semantic_started) * 1000
            if ranked is not None:
                selected_paths = set(ranked)
                selected_soft = tuple(
                    item for item in bounded_soft if item.path in selected_paths
                )
                semantic_used = True
        selected = tuple(sorted((*hard, *selected_soft), key=_candidate_sort_key))
        context_slices = self._context_slices(selected, snapshot)
        context_bytes = sum(item.bytes_selected for item in context_slices)
        selected_bytes = sum(item.size for item in selected)
        allowed, forbidden = self._mutation_scope(requirements, snapshot)
        metrics = CandidateGenerationMetrics(
            round((time.perf_counter() - started) * 1000, 3),
            round(semantic_ms, 3),
            len(candidates),
            len(selected),
            selected_bytes,
            context_bytes,
            sum(item.estimated_tokens for item in context_slices),
            context_bytes > self.budget.max_context_bytes,
            sum(item.path not in {value.path for value in selected} for item in hard),
            semantic_used,
        )
        return ProjectCandidateSelection(
            snapshot.project_path,
            snapshot.project_id,
            tuple(candidates),
            selected,
            context_slices,
            tuple(item.path for item in selected),
            allowed,
            forbidden,
            metrics,
        )

    @staticmethod
    def _requirements_payload(requirements: Any | None) -> dict[str, Any]:
        if requirements is None:
            return {}
        if hasattr(requirements, "as_dict"):
            return requirements.as_dict()
        if isinstance(requirements, Mapping):
            return dict(requirements)
        return {}

    def _explicit_files(
        self,
        task: str,
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
    ) -> None:
        by_basename: dict[str, list[str]] = {}
        explicit_paths: set[str] = set()
        for item in snapshot.files:
            by_basename.setdefault(Path(item.path).name.casefold(), []).append(item.path)
            escaped = re.escape(item.path.casefold())
            if re.search(rf"(?<![\w./]){escaped}(?![\w./])", task):
                explicit_paths.add(item.path)
                aggregator.add(
                    item.path,
                    RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE,
                    EvidenceStrength.HARD,
                    item.path,
                    "exact_repository_path",
                    "task:explicit_file_path",
                )
        for basename, paths in by_basename.items():
            if re.search(rf"(?<![\w.]){re.escape(basename)}(?![\w.])", task):
                full_matches = [path for path in paths if path in explicit_paths]
                if full_matches:
                    paths = full_matches
                for path in paths:
                    aggregator.add(
                        path,
                        RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE,
                        EvidenceStrength.HARD,
                        basename,
                        "exact_basename",
                        "task:explicit_file_basename",
                    )
        for directory in sorted(snapshot.directories, key=len, reverse=True):
            normalized = directory.casefold().rstrip("/")
            if "/" not in normalized:
                continue
            if re.search(rf"(?<![\w./]){re.escape(normalized)}/?(?![/\w.])", task):
                for item in snapshot.files:
                    if item.path.startswith(f"{directory}/"):
                        aggregator.add(
                            item.path,
                            RelevantFileEvidenceSource.EXPLICIT_DIRECTORY_REFERENCE,
                            EvidenceStrength.STRONG,
                            directory,
                            "explicit_directory_member",
                            "task:explicit_directory",
                        )

    def _tracebacks(
        self,
        task: str,
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
    ) -> None:
        for pattern in (self._TRACEBACK, self._PATH_LINE):
            for match in pattern.finditer(task):
                path = self._resolve_context_path(match.group("path"), snapshot)
                if path:
                    aggregator.add(
                        path,
                        RelevantFileEvidenceSource.TRACEBACK_REFERENCE,
                        EvidenceStrength.HARD,
                        f"{path}:{match.group('line')}",
                        "exact_traceback_location",
                        "task:traceback_path_line",
                    )

    @staticmethod
    def _resolve_context_path(raw_path: str, snapshot: ProjectSnapshotV2) -> str | None:
        normalized = raw_path.replace("\\", "/")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=False)
                relative = resolved.relative_to(Path(snapshot.project_path)).as_posix()
                return relative if relative in snapshot.file_index else None
            except ValueError:
                return None
        normalized = normalized.removeprefix("./")
        return normalized if normalized in snapshot.file_index else None

    def _symbols(
        self,
        task: str,
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
    ) -> None:
        index = snapshot.symbol_index
        for match in self._IDENTIFIER.finditer(task):
            token = match.group(1)
            explicitly_code_shaped = bool(
                "_" in token
                or "." in token
                or any(character.isupper() for character in token[1:])
                or (token.isupper() and len(token) >= 4)
                or f"`{token}`" in task
                or f"{token}()" in task
            )
            if not explicitly_code_shaped:
                continue
            records = index.get(token, ())
            if not records and "." in token:
                records = index.get(token.rsplit(".", 1)[-1], ())
            for symbol in records:
                aggregator.add(
                    symbol.file,
                    RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE,
                    EvidenceStrength.HARD,
                    token,
                    "exact_symbol_token",
                    "task:explicit_symbol",
                )
                aggregator.add(
                    symbol.file,
                    RelevantFileEvidenceSource.SYMBOL_DEFINITION,
                    EvidenceStrength.STRONG,
                    symbol.qualified_name,
                    f"definition:{symbol.kind.value}:{symbol.line}",
                    "index:symbol_definition",
                )
                if symbol.name.endswith(("Error", "Exception")):
                    aggregator.add(
                        symbol.file,
                        RelevantFileEvidenceSource.ERROR_REFERENCE,
                        EvidenceStrength.STRONG,
                        symbol.name,
                        "exception_definition",
                        "index:error_symbol",
                    )

    def _structure(
        self,
        task: str,
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
    ) -> None:
        tokens = {item.casefold() for item in self._TASK_TOKEN.findall(task)}
        ignored = {
            "file",
            "line",
            "main",
            "module",
            "orchestrator",
            "project",
            "python",
            "test",
            "tests",
            "tern",
        }
        tokens -= ignored
        for item in snapshot.files:
            if item.language not in {
                "Python",
                "JavaScript",
                "TypeScript",
                "HTML",
                "CSS",
                "C",
                "C++",
                "Rust",
                "Go",
                "Java",
            }:
                continue
            structural = set(
                token
                for token in re.split(
                    r"[^A-Za-z0-9]+", Path(item.path).stem.casefold()
                )
                if len(token) >= 4 and token not in ignored
            )
            matched = sorted(tokens & structural)
            if matched:
                aggregator.add(
                    item.path,
                    RelevantFileEvidenceSource.PROJECT_STRUCTURE,
                    EvidenceStrength.SUPPORTING,
                    ",".join(matched),
                    "path_token_match",
                    "index:project_structure",
                )

    def _relationships(
        self,
        seeds: Sequence[str],
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
        *,
        requirements: Any | None,
    ) -> None:
        seed_set = set(seeds)
        seed_contains_test = any(snapshot.file_index[path].is_test for path in seeds)
        tests_required = self._requires_true(requirements, "test_execution")
        for path in seeds:
            for dependency in snapshot.import_graph.get(path, ()):
                aggregator.add(
                    dependency,
                    RelevantFileEvidenceSource.IMPORT_DEPENDENCY,
                    EvidenceStrength.STRONG,
                    path,
                    "direct_import",
                    "index:import_graph",
                )
            for importer in snapshot.reverse_import_graph.get(path, ()):
                if snapshot.file_index[importer].is_test and not (
                    tests_required or seed_contains_test
                ):
                    continue
                aggregator.add(
                    importer,
                    RelevantFileEvidenceSource.REVERSE_IMPORT,
                    EvidenceStrength.STRONG,
                    path,
                    "direct_reverse_import",
                    "index:reverse_import_graph",
                )
        for relation in snapshot.test_relationships:
            if not (tests_required or seed_contains_test):
                continue
            if relation.production_file in seed_set or relation.test_file in seed_set:
                target = (
                    relation.test_file
                    if relation.production_file in seed_set
                    else relation.production_file
                )
                strength = (
                    EvidenceStrength.STRONG
                    if tests_required or relation.strength is EvidenceStrength.STRONG
                    else EvidenceStrength.SUPPORTING
                )
                aggregator.add(
                    target,
                    RelevantFileEvidenceSource.TEST_RELATIONSHIP,
                    strength,
                    relation.production_file,
                    "+".join(relation.evidence),
                    "index:test_relationship",
                )
        for path in tuple(aggregator.values):
            item = snapshot.file_index[path]
            if item.is_config:
                aggregator.add(
                    path,
                    RelevantFileEvidenceSource.CONFIG_RELATIONSHIP,
                    EvidenceStrength.SUPPORTING,
                    path,
                    "known_config_file",
                    "index:config_file",
                )
            if item.is_entrypoint:
                aggregator.add(
                    path,
                    RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP,
                    EvidenceStrength.STRONG,
                    path,
                    "declared_entrypoint",
                    "index:entrypoint",
                )
            if Path(path).name.casefold() == "pyproject.toml":
                for entrypoint in snapshot.declared_entry_points:
                    aggregator.add(
                        entrypoint,
                        RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP,
                        EvidenceStrength.STRONG,
                        path,
                        "declared_by_pyproject",
                        "index:pyproject_entrypoint",
                    )
            elif item.is_entrypoint and "pyproject.toml" in snapshot.file_index:
                aggregator.add(
                    "pyproject.toml",
                    RelevantFileEvidenceSource.CONFIG_RELATIONSHIP,
                    EvidenceStrength.SUPPORTING,
                    path,
                    "entrypoint_declaration",
                    "index:pyproject_entrypoint",
                )

    @staticmethod
    def _requires_true(requirements: Any | None, name: str) -> bool:
        if requirements is None:
            return False
        values = getattr(requirements, "requirements", None)
        if isinstance(values, Mapping) and name in values:
            value = getattr(values[name], "value", None)
            return getattr(value, "value", value) == "TRUE"
        if isinstance(requirements, Mapping):
            value = requirements.get(name)
            if isinstance(value, Mapping):
                value = value.get("value")
            return value is True or value == "TRUE"
        return False

    @staticmethod
    def _git_evidence(
        snapshot: ProjectSnapshotV2,
        aggregator: _EvidenceAggregator,
    ) -> None:
        changed = set(snapshot.modified_files) | set(snapshot.untracked_files)
        diff = set(snapshot.diff_files)
        for path in tuple(aggregator.values):
            if path in changed:
                aggregator.add(
                    path,
                    RelevantFileEvidenceSource.GIT_MODIFIED_FILE,
                    EvidenceStrength.SUPPORTING,
                    path,
                    "candidate_is_modified",
                    "git:status",
                )
            if path in diff:
                aggregator.add(
                    path,
                    RelevantFileEvidenceSource.GIT_DIFF_RELATIONSHIP,
                    EvidenceStrength.SUPPORTING,
                    path,
                    "candidate_has_diff",
                    "git:diff_name",
                )

    def _context_slices(
        self,
        selected: Sequence[RelevantFileCandidate],
        snapshot: ProjectSnapshotV2,
    ) -> tuple[ContextSlice, ...]:
        remaining = self.budget.max_context_bytes
        slices: list[ContextSlice] = []
        root = Path(snapshot.project_path)
        for candidate in selected:
            if remaining <= 0:
                break
            indexed = snapshot.file_index[candidate.path]
            max_bytes = min(self.budget.max_context_bytes_per_file, remaining)
            start_line = 1
            end_line = 1
            evidence_ref = candidate.evidences[0].evidence_ref
            symbol_targets = {
                evidence.target
                for evidence in candidate.evidences
                if evidence.source
                in {
                    RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE,
                    RelevantFileEvidenceSource.SYMBOL_DEFINITION,
                }
            }
            matching = [
                symbol
                for symbol in indexed.symbols
                if symbol.name in symbol_targets or symbol.qualified_name in symbol_targets
            ]
            if matching:
                symbol = min(matching, key=lambda item: item.line)
                start_line = symbol.line
                end_line = min(
                    symbol.end_line,
                    symbol.line + self.budget.max_symbol_lines - 1,
                )
            raw_lines = (root / candidate.path).read_bytes().splitlines(keepends=True)
            if not raw_lines:
                selected_bytes = 0
                end_line = 0
            else:
                start_index = min(max(start_line - 1, 0), len(raw_lines) - 1)
                if not matching:
                    end_line = len(raw_lines)
                chunks: list[bytes] = []
                for line in raw_lines[start_index:end_line]:
                    if sum(len(item) for item in chunks) + len(line) > max_bytes:
                        break
                    chunks.append(line)
                selected_bytes = sum(len(item) for item in chunks)
                end_line = start_line + max(0, len(chunks) - 1)
            slices.append(
                ContextSlice(
                    candidate.path,
                    start_line,
                    end_line,
                    selected_bytes,
                    math.ceil(selected_bytes / 4),
                    evidence_ref,
                )
            )
            remaining -= selected_bytes
        return tuple(slices)

    @staticmethod
    def _mutation_scope(
        requirements: Any | None,
        snapshot: ProjectSnapshotV2,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        expected = tuple(getattr(requirements, "expected_files", ()) or ())
        forbidden = tuple(getattr(requirements, "forbidden_files", ()) or ())
        known = snapshot.file_index
        return (
            tuple(path for path in expected if path in known),
            tuple(path for path in forbidden if path in known or path.startswith("scope:")),
        )
