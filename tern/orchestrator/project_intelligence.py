from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LANGUAGES = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".css": "CSS",
    ".c": "C",
    ".cpp": "C++",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
}
CONFIG_FILES = {
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    ".gitignore",
    ".gitattributes",
}
DEPENDENCY_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.toml",
    "go.mod",
}
IMPORTANT_DIRECTORIES = {"src", "lib", "app", "tests", "test", "docs", "scripts", "tern"}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".orchestrator",
    "models",
}


@dataclass(frozen=True)
class ExplorationBudget:
    max_files_per_analysis: int = 80
    max_bytes_per_file: int = 65_536
    max_total_context_bytes: int = 524_288
    max_repo_entries: int = 10_000


@dataclass(frozen=True)
class RepoMapEntry:
    path: str
    file_type: str
    module: str | None
    symbols: tuple[str, ...]
    imports: tuple[str, ...]
    size: int
    modified_ns: int
    sha256: str | None
    analyzed: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RepoMapEntry":
        return cls(
            str(value["path"]),
            str(value["file_type"]),
            value.get("module"),
            tuple(value.get("symbols") or ()),
            tuple(value.get("imports") or ()),
            int(value["size"]),
            int(value["modified_ns"]),
            value.get("sha256"),
            bool(value.get("analyzed")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_type": self.file_type,
            "module": self.module,
            "symbols": list(self.symbols),
            "imports": list(self.imports),
            "size": self.size,
            "modified_ns": self.modified_ns,
            "sha256": self.sha256,
            "analyzed": self.analyzed,
        }


@dataclass(frozen=True)
class ProjectSnapshot:
    project_path: str
    languages: tuple[str, ...]
    important_directories: tuple[str, ...]
    important_files: tuple[str, ...]
    modules: tuple[str, ...]
    entry_points: tuple[str, ...]
    tests: tuple[str, ...]
    configuration_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    git_branch: str | None
    git_status: str
    modified_files: tuple[str, ...]
    untracked_files: tuple[str, ...]
    recent_relevant_changes: tuple[str, ...]
    known_test_state: str | None
    known_failures: tuple[str, ...]
    repo_map: tuple[RepoMapEntry, ...]
    budget: ExplorationBudget
    files_reused: int
    files_analyzed: int
    bytes_analyzed: int
    truncated: bool

    def compact(
        self,
        *,
        max_entries: int = 80,
        max_context_bytes: int = 24_000,
    ) -> dict[str, Any]:
        limit = lambda values: list(values[:80])
        payload: dict[str, Any] = {
            "project_path": self.project_path,
            "languages": list(self.languages),
            "important_directories": list(self.important_directories),
            "important_files": limit(self.important_files),
            "modules": limit(self.modules),
            "entry_points": limit(self.entry_points),
            "tests": limit(self.tests),
            "configuration_files": limit(self.configuration_files),
            "dependency_files": limit(self.dependency_files),
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "modified_files": limit(self.modified_files),
            "untracked_files": limit(self.untracked_files),
            "recent_relevant_changes": list(self.recent_relevant_changes),
            "known_test_state": self.known_test_state,
            "known_failures": limit(self.known_failures),
            "repo_map": [],
            "repo_map_total": len(self.repo_map),
            "budget": self.budget.__dict__,
            "truncated": self.truncated or len(self.repo_map) > max_entries,
        }
        modified = set(self.modified_files) | set(self.untracked_files)

        source_entries = sorted(
            (
                item
                for item in self.repo_map
                if item.path.startswith("tern/") and item.file_type == "Python"
            ),
            key=lambda item: item.path.casefold(),
        )
        test_entries = sorted(
            (
                item
                for item in self.repo_map
                if item.path.startswith("tests/") and item.file_type == "Python"
            ),
            key=lambda item: item.path.casefold(),
        )
        source_and_tests = set(source_entries) | set(test_entries)
        ordered_entries: list[RepoMapEntry] = []
        source_index = test_index = 0
        while source_index < len(source_entries) or test_index < len(test_entries):
            for _ in range(2):
                if source_index < len(source_entries):
                    ordered_entries.append(source_entries[source_index])
                    source_index += 1
            if test_index < len(test_entries):
                ordered_entries.append(test_entries[test_index])
                test_index += 1
        ordered_entries.extend(
            sorted(
                (item for item in self.repo_map if item not in source_and_tests),
                key=lambda item: (
                    item.path not in modified,
                    item.path not in self.important_files,
                    item.path.casefold(),
                ),
            )
        )
        for item in ordered_entries[:max_entries]:
            compact_entry = {
                "path": item.path,
                "file_type": item.file_type,
                "module": item.module,
                "symbols": list(item.symbols[:6]),
                "imports": list(item.imports[:6]),
                "size": item.size,
                "analyzed": item.analyzed,
            }
            payload["repo_map"].append(compact_entry)
            if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_context_bytes:
                payload["repo_map"].pop()
                payload["truncated"] = True
                break
        return payload


class ProjectSnapshotBuilder:
    def __init__(
        self,
        project_path: str | Path,
        *,
        cache_path: str | Path | None = None,
        budget: ExplorationBudget | None = None,
    ) -> None:
        self.root = Path(project_path).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("project_path must be a directory")
        self.cache_path = Path(cache_path) if cache_path else None
        self.budget = budget or ExplorationBudget()

    def build(
        self,
        *,
        known_test_state: str | None = None,
        known_failures: Iterable[str] = (),
    ) -> ProjectSnapshot:
        paths = self._project_files()
        previous = self._load_cache()
        entries: list[RepoMapEntry] = []
        reused = analyzed = bytes_analyzed = 0
        for path in paths:
            relative = path.relative_to(self.root).as_posix()
            stat = path.stat()
            old = previous.get(relative)
            if old and old.size == stat.st_size and old.modified_ns == stat.st_mtime_ns:
                entries.append(old)
                reused += 1
                continue
            can_analyze = (
                analyzed < self.budget.max_files_per_analysis
                and stat.st_size <= self.budget.max_bytes_per_file
                and bytes_analyzed + stat.st_size <= self.budget.max_total_context_bytes
            )
            entry = self._map_entry(path, relative, stat, analyze=can_analyze)
            entries.append(entry)
            if entry.analyzed:
                analyzed += 1
                bytes_analyzed += stat.st_size

        entries.sort(key=lambda item: item.path.casefold())
        self._save_cache(entries)
        status = self._git_status()
        relative_paths = {item.path for item in entries}
        top_directories = sorted(
            {
                item.path.split("/", 1)[0]
                for item in entries
                if "/" in item.path
                and item.path.split("/", 1)[0].casefold() in IMPORTANT_DIRECTORIES
            }
        )
        important_files = sorted(
            path for path in relative_paths if Path(path).name.casefold() in CONFIG_FILES | DEPENDENCY_FILES | {"readme.md"}
        )
        tests = sorted(
            path
            for path in relative_paths
            if path.startswith(("tests/", "test/")) or Path(path).name.startswith("test_")
        )
        modules = sorted({item.module for item in entries if item.module})
        entry_points = sorted(self._entry_points(relative_paths))
        return ProjectSnapshot(
            str(self.root),
            tuple(sorted({LANGUAGES[Path(item.path).suffix.casefold()] for item in entries if Path(item.path).suffix.casefold() in LANGUAGES})),
            tuple(top_directories),
            tuple(important_files),
            tuple(modules),
            tuple(entry_points),
            tuple(tests),
            tuple(sorted(path for path in relative_paths if Path(path).name.casefold() in CONFIG_FILES)),
            tuple(sorted(path for path in relative_paths if Path(path).name.casefold() in DEPENDENCY_FILES)),
            status["branch"],
            status["summary"],
            tuple(status["modified"]),
            tuple(status["untracked"]),
            tuple(self._recent_changes()),
            known_test_state,
            tuple(known_failures),
            tuple(entries),
            self.budget,
            reused,
            analyzed,
            bytes_analyzed,
            len(paths) >= self.budget.max_repo_entries,
        )

    def _project_files(self) -> list[Path]:
        if (self.root / ".git").exists():
            completed = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                names = [item for item in completed.stdout.decode("utf-8", errors="replace").split("\0") if item]
                result = []
                for name in names[: self.budget.max_repo_entries]:
                    path = (self.root / name).resolve()
                    if path.is_file() and self.root in path.parents:
                        result.append(path)
                return sorted(set(result), key=lambda item: str(item).casefold())
        result: list[Path] = []
        for current, directories, files in os.walk(self.root, followlinks=False):
            directories[:] = [name for name in directories if name.casefold() not in IGNORED_DIRECTORIES]
            for name in files:
                path = Path(current, name)
                if path.is_symlink():
                    continue
                result.append(path)
                if len(result) >= self.budget.max_repo_entries:
                    return result
        return result

    def _map_entry(self, path: Path, relative: str, stat: os.stat_result, *, analyze: bool) -> RepoMapEntry:
        suffix = path.suffix.casefold()
        module = None
        symbols: tuple[str, ...] = ()
        imports: tuple[str, ...] = ()
        digest = None
        if suffix == ".py":
            module = relative[:-3].replace("/", ".").removesuffix(".__init__")
        if analyze:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if suffix == ".py":
                try:
                    tree = ast.parse(raw.decode("utf-8", errors="replace"), filename=relative)
                    symbols = tuple(
                        node.name
                        for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    )[:80]
                    found_imports: list[str] = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            found_imports.extend(alias.name for alias in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            found_imports.append(node.module)
                    imports = tuple(dict.fromkeys(found_imports))[:80]
                except SyntaxError:
                    pass
        return RepoMapEntry(
            relative,
            LANGUAGES.get(suffix, suffix.lstrip(".") or "file"),
            module,
            symbols,
            imports,
            stat.st_size,
            stat.st_mtime_ns,
            digest,
            analyze,
        )

    def _entry_points(self, paths: set[str]) -> set[str]:
        result = {path for path in paths if Path(path).name in {"main.py", "__main__.py"}}
        pyproject = self.root / "pyproject.toml"
        if pyproject.is_file():
            try:
                value = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                for name, target in value.get("project", {}).get("scripts", {}).items():
                    result.add(f"pyproject:{name}={target}")
            except (OSError, tomllib.TOMLDecodeError):
                pass
        return result

    def _git_status(self) -> dict[str, Any]:
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
            records = [item for item in completed.stdout.decode("utf-8", errors="replace").split("\0") if item]
            for record in records:
                code, path = record[:2], record[3:]
                (untracked if code == "??" else modified).append(path)
        summary = "clean" if not modified and not untracked else "dirty"
        return {"branch": branch, "summary": summary, "modified": sorted(modified), "untracked": sorted(untracked)}

    def _recent_changes(self) -> list[str]:
        completed = subprocess.run(
            ["git", "-C", str(self.root), "log", "-5", "--pretty=format:%h %s"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return [line for line in completed.stdout.splitlines() if line][:5]

    def _load_cache(self) -> dict[str, RepoMapEntry]:
        if self.cache_path is None:
            return {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if value.get("version") != 1 or value.get("project_path") != str(self.root):
                return {}
            return {item["path"]: RepoMapEntry.from_dict(item) for item in value.get("entries", [])}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return {}

    def _save_cache(self, entries: list[RepoMapEntry]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "project_path": str(self.root),
            "entries": [item.as_dict() for item in entries],
        }
        handle, temporary_name = tempfile.mkstemp(prefix="snapshot-", suffix=".json", dir=self.cache_path.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProjectFileSelectionResult:
    selected_files: tuple[str, ...]
    probable_modules: tuple[str, ...]
    valid: bool
    latency_ms: float
    error_code: str | None = None


def project_file_selection_schema() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_files": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
                "maxItems": 20,
            },
            "probable_modules": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
                "maxItems": 12,
            },
        },
        "required": ["selected_files", "probable_modules"],
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "project_file_selection", "strict": True, "schema": schema},
    }


class ProjectFileSelector:
    """Read-only relevance proposal over a bounded ProjectSnapshot."""

    system_prompt = (
        "Select only repository files likely relevant to understand the task. "
        "Use paths present in the project snapshot. Do not propose edits, execute "
        "tools, or select an agent. Prefer a small sufficient set."
    )

    def __init__(self, client: Any):
        self.client = client

    def select(self, task: str, snapshot: dict[str, Any]) -> ProjectFileSelectionResult:
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"task": task, "project_snapshot": snapshot}, ensure_ascii=False
                ),
            },
        ]
        try:
            response = self.client.chat(
                messages,
                response_format=project_file_selection_schema(),
                temperature=0.0,
                max_tokens=256,
            )
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content)
            known_paths = {
                str(item.get("path"))
                for item in snapshot.get("repo_map", [])
                if item.get("path")
            }
            known_paths.update(snapshot.get("tests") or ())
            known_paths.update(snapshot.get("important_files") or ())
            selected = tuple(str(item) for item in value["selected_files"])
            if not all(path in known_paths for path in selected):
                raise ValueError("selection contains path outside snapshot")
            modules = tuple(str(item) for item in value["probable_modules"])
            return ProjectFileSelectionResult(
                selected,
                modules,
                True,
                round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:
            return ProjectFileSelectionResult(
                (),
                (),
                False,
                round((time.perf_counter() - started) * 1000, 3),
                type(exc).__name__,
            )
