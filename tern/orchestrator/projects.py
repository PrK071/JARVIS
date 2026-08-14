from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .codex_state import FileMutex
from .security import AccessDenied, PathPolicy


PROJECT_MARKERS = {
    ".git": 4,
    "pyproject.toml": 3,
    "package.json": 3,
    "build.gradle": 3,
    "settings.gradle": 3,
    "Cargo.toml": 3,
    "CMakeLists.txt": 3,
    "pom.xml": 3,
    "requirements.txt": 2,
    "README.md": 1,
    "src": 1,
    "tests": 1,
}

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".orchestrator",
        ".venv",
        "node_modules",
        "__pycache__",
        "build",
        "dist",
        "target",
        "models",
        "checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
    }
)
IGNORED_SUFFIXES = frozenset(
    {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".onnx",
        ".gguf",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".pyc",
        ".pyd",
        ".dll",
        ".exe",
    }
)
SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "credentials.json",
        "secrets.json",
    }
)

DEFAULT_ALIASES = {
    "tern": ("tern", "jarvis", "assistente", "orquestrador"),
    "llama.cpp": ("llama.cpp", "llama", "lama ponto cpp"),
    "sasori_review": ("sasori_review", "sasori", "sasori review"),
}
DEFAULT_NAMES = {
    "tern": "Tern",
    "llama.cpp": "llama.cpp",
    "sasori_review": "Sasori Review",
}

DESCRIPTION_TERMS = {
    "daniel": ("voice", "windows", "speech", "tts", "config", "env"),
    "voz": ("voice", "tts", "speech", "audio", "config"),
    "bridge": ("codex", "bridge", "collaboration"),
    "codex": ("codex", "bridge"),
    "confirmacao": ("policy", "pending", "confirmation"),
    "politica": ("policy", "security"),
    "jobs": ("codex_jobs", "jobs"),
    "assincronos": ("codex_jobs", "jobs"),
    "configuracao": ("config", "settings", "env"),
    "testes": ("test", "tests"),
    "documentacao": ("docs", "readme"),
    "provider": ("provider", "tts", "speech"),
}
STOP_WORDS = frozenset(
    {
        "a",
        "as",
        "de",
        "da",
        "das",
        "do",
        "dos",
        "e",
        "o",
        "os",
        "um",
        "uma",
        "para",
        "por",
        "aquele",
        "aquela",
        "arquivo",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9._+ -]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_technical_transcript(value: str) -> str:
    """Correct known technical names without globally rewriting ordinary speech."""
    technical = bool(
        re.search(
            r"(?i)\b(?:arquivo|codigo|c[oó]digo|projeto|assistente|jarves|jarvis|"
            r"config|diretorio|pasta|sessao|thread|teste|bridge|voz|provider|"
            r"codex|c[oó]dex|c[oó]digo ex|terne|ll?ama|lama ponto cpp|"
            r"dip\s+(?:sique|chique)|acessao|uornings)\b",
            value,
        )
    )
    if not technical:
        return value
    replacements = (
        (r"(?i)\bc[oó]digo\s+ex\b|\bc[oó]dex\b", "Codex"),
        (r"(?i)\bterne\b", "Tern"),
        (r"(?i)\blama\s+ponto\s+cpp\b", "llama.cpp"),
        (r"(?i)\bjarves\b", "Jarvis"),
        (r"(?i)\bdip\s+(?:sique|chique)\b", "DeepSeek"),
        (r"(?i)\bacessao\b", "sessao"),
        (r"(?i)\buornings\b", "warnings"),
    )
    result = value
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


def _project_id(root: Path) -> str:
    value = normalize_name(root.name).replace(" ", "-")
    return value or f"project-{uuid.uuid4().hex[:8]}"


def _project_type(markers: Iterable[str]) -> str:
    values = set(markers)
    if "CMakeLists.txt" in values:
        return "cmake"
    if "pyproject.toml" in values or "requirements.txt" in values:
        return "python"
    if "package.json" in values:
        return "javascript"
    if "Cargo.toml" in values:
        return "rust"
    if {"pom.xml", "build.gradle", "settings.gradle"}.intersection(values):
        return "jvm"
    return "unknown"


class ProjectRegistry:
    def __init__(
        self,
        policy: PathPolicy,
        state_dir: Path,
        *,
        codex: Any | None = None,
    ):
        self.policy = policy
        self.state_dir = state_dir
        self.codex = codex
        self.path = state_dir / "projects.json"
        self.lock_path = state_dir / "projects.lock"
        self.index_dir = state_dir / "project-indexes"
        self._ensure_registry()

    def _default(self) -> dict[str, Any]:
        return {
            "version": 1,
            "projects": [],
            "active_project_id": None,
            "last_tool_project_id": None,
            "recent_files": [],
            "updated_at": utc_now(),
        }

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("projects"), list):
                raise ValueError("registro invalido")
            return value
        except (OSError, json.JSONDecodeError, ValueError):
            return self._default()

    def _write_unlocked(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + os.linesep,
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def read(self) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            return self._read_unlocked()

    def _ensure_registry(self) -> None:
        value = self.read()
        if not value["projects"]:
            self.refresh()

    def _markers(self, root: Path) -> tuple[list[str], int]:
        markers: list[str] = []
        score = 0
        for marker, points in PROJECT_MARKERS.items():
            candidate = root / marker
            try:
                if candidate.exists() and not candidate.is_symlink():
                    self.policy.resolve(str(candidate))
                    markers.append(marker)
                    score += points
            except (OSError, AccessDenied):
                continue
        return markers, score

    def _candidate_directories(self, max_depth: int) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()
        for configured_root in self.policy.roots:
            try:
                root = self.policy.resolve(str(configured_root))
            except (OSError, AccessDenied):
                continue
            queue = [(root, 0)]
            while queue:
                current, depth = queue.pop(0)
                key = str(current).casefold()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(current)
                if depth >= max_depth:
                    continue
                try:
                    children = sorted(current.iterdir(), key=lambda item: item.name.casefold())
                except OSError:
                    continue
                for child in children:
                    if (
                        child.name.casefold() in IGNORED_DIRECTORIES
                        or child.name.casefold().startswith(".venv-")
                        or child.name.casefold().startswith("pytest-")
                        or child.is_symlink()
                    ):
                        continue
                    try:
                        if child.is_dir():
                            self.policy.resolve(str(child))
                            queue.append((child, depth + 1))
                    except (OSError, AccessDenied):
                        continue
        return candidates

    def refresh(self, *, max_depth: int = 2) -> dict[str, Any]:
        existing = self.read()
        existing_by_root = {
            str(Path(item["root"]).resolve()).casefold(): item
            for item in existing.get("projects", [])
            if isinstance(item, dict) and item.get("root")
        }
        configured = {
            str(Path(root).resolve()).casefold() for root in self.policy.roots if Path(root).exists()
        }
        projects: list[dict[str, Any]] = []
        roots_seen: set[str] = set()
        for candidate in self._candidate_directories(max_depth):
            markers, score = self._markers(candidate)
            root_key = str(candidate.resolve()).casefold()
            if root_key not in configured and (
                ".git" not in markers or len(markers) < 2 or score < 5
            ):
                continue
            old = existing_by_root.get(root_key, {})
            project_id = str(old.get("id") or _project_id(candidate))
            aliases = {
                normalize_name(alias)
                for alias in old.get("aliases", [])
                if normalize_name(str(alias))
            }
            aliases.add(normalize_name(candidate.name))
            aliases.update(DEFAULT_ALIASES.get(project_id, ()))
            aliases = {normalize_name(alias) for alias in aliases if normalize_name(alias)}
            projects.append(
                {
                    "id": project_id,
                    "name": DEFAULT_NAMES.get(
                        project_id,
                        str(old.get("name") or candidate.name),
                    ),
                    "root": str(candidate.resolve()),
                    "aliases": sorted(aliases),
                    "type": _project_type(markers),
                    "markers": sorted(markers),
                    "last_used_at": old.get("last_used_at"),
                    "source": old.get("source") or (
                        "configured" if root_key in configured else "discovered"
                    ),
                }
            )
            roots_seen.add(root_key)
        active = existing.get("active_project_id")
        ids = {item["id"] for item in projects}
        if active not in ids:
            tern = next((item for item in projects if item["id"] == "tern"), None)
            active = (tern or (projects[0] if projects else {})).get("id")
        value = {
            **self._default(),
            "projects": sorted(projects, key=lambda item: item["name"].casefold()),
            "active_project_id": active,
            "last_tool_project_id": (
                existing.get("last_tool_project_id")
                if existing.get("last_tool_project_id") in ids
                else None
            ),
            "recent_files": [
                item
                for item in existing.get("recent_files", [])
                if isinstance(item, dict) and item.get("project_id") in ids
            ][-12:],
            "updated_at": utc_now(),
        }
        with FileMutex(self.lock_path):
            self._write_unlocked(value)
        return value

    def projects(self) -> list[dict[str, Any]]:
        value = self.read()
        valid: list[dict[str, Any]] = []
        stale = False
        for project in value["projects"]:
            try:
                root = self.policy.resolve(str(project["root"]))
                if not root.is_dir():
                    stale = True
                    continue
            except (OSError, AccessDenied):
                stale = True
                continue
            valid.append(dict(project))
        return self.refresh()["projects"] if stale else valid

    def _shared_project(self) -> Path | None:
        shared = getattr(self.codex, "shared_project", None)
        if callable(shared):
            try:
                value = shared()
                return self.policy.resolve(str(value)) if value else None
            except Exception:
                return None
        return None

    @staticmethod
    def _contains_alias(query: str, alias: str) -> bool:
        if not alias:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", query) is not None

    def resolve(
        self,
        *,
        query: str | None = None,
        path_hint: str | None = None,
        require_unique: bool = True,
        use_fallbacks: bool = True,
    ) -> dict[str, Any]:
        projects = self.projects()
        if path_hint:
            try:
                hinted = self.policy.resolve(path_hint)
            except Exception as exc:
                return {"ok": False, "error": "project_path_not_allowed", "message": str(exc)}
            matches = [
                item
                for item in projects
                if hinted == Path(item["root"]) or Path(item["root"]) in hinted.parents
            ]
            if matches:
                return self._resolution(matches[0], 1.0, "path", [])
            return {"ok": False, "error": "project_not_registered", "path": str(hinted)}

        normalized_query = normalize_name(query or "")
        matches: list[tuple[dict[str, Any], str]] = []
        for project in projects:
            for alias in project.get("aliases", []):
                if self._contains_alias(normalized_query, normalize_name(alias)):
                    matches.append((project, "alias"))
                    break
            else:
                name = normalize_name(str(project.get("name") or ""))
                if self._contains_alias(normalized_query, name):
                    matches.append((project, "name"))
        unique = {item["id"]: (item, matched_by) for item, matched_by in matches}
        if len(unique) == 1:
            project, matched_by = next(iter(unique.values()))
            return self._resolution(project, 0.98, matched_by, [])
        if len(unique) > 1:
            alternatives = [self._brief(item) for item, _ in unique.values()]
            if require_unique:
                return {"ok": False, "error": "ambiguous_project", "alternatives": alternatives}
            project, matched_by = next(iter(unique.values()))
            return self._resolution(project, 0.65, matched_by, alternatives[1:])
        if not use_fallbacks:
            return {"ok": False, "error": "project_not_found", "alternatives": []}

        shared = self._shared_project()
        if shared:
            project = next((item for item in projects if Path(item["root"]) == shared), None)
            if project:
                return self._resolution(project, 0.93, "codex_thread", [])
        state = self.read()
        for field, matched_by, confidence in (
            ("last_tool_project_id", "last_tool", 0.90),
            ("active_project_id", "active", 0.88),
        ):
            project = next((item for item in projects if item["id"] == state.get(field)), None)
            if project:
                return self._resolution(project, confidence, matched_by, [])
        try:
            cwd = self.policy.resolve(os.getcwd())
        except Exception:
            cwd = None
        if cwd:
            project = next(
                (item for item in projects if cwd == Path(item["root"]) or Path(item["root"]) in cwd.parents),
                None,
            )
            if project:
                return self._resolution(project, 0.80, "working_directory", [])
        return {"ok": False, "error": "project_ambiguous", "alternatives": [self._brief(item) for item in projects]}

    @staticmethod
    def _brief(project: dict[str, Any]) -> dict[str, Any]:
        return {"project_id": project["id"], "name": project["name"], "root": project["root"]}

    def _resolution(
        self,
        project: dict[str, Any],
        confidence: float,
        matched_by: str,
        alternatives: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.use(str(project["id"]), matched_by=matched_by)
        return {
            "ok": True,
            "project_id": project["id"],
            "name": project["name"],
            "root": project["root"],
            "confidence": confidence,
            "matched_by": matched_by,
            "alternatives": alternatives,
            "error": None,
        }

    def use(self, query: str, *, matched_by: str = "explicit") -> dict[str, Any]:
        projects = self.projects()
        normalized = normalize_name(query)
        project = next(
            (
                item
                for item in projects
                if item["id"] == query
                or normalize_name(item["name"]) == normalized
                or normalized in {normalize_name(alias) for alias in item.get("aliases", [])}
                or str(Path(item["root"])).casefold() == str(Path(query)).casefold()
            ),
            None,
        )
        if project is None:
            result = self.resolve(query=query, path_hint=query if Path(query).is_absolute() else None, use_fallbacks=False)
            if not result.get("ok"):
                return result
            project = next(item for item in projects if item["id"] == result["project_id"])
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            state["active_project_id"] = project["id"]
            state["last_tool_project_id"] = project["id"]
            state["updated_at"] = utc_now()
            for item in state["projects"]:
                if item["id"] == project["id"]:
                    item["last_used_at"] = state["updated_at"]
            self._write_unlocked(state)
        return self._resolution_data(project, 1.0, matched_by)

    @staticmethod
    def _resolution_data(project: dict[str, Any], confidence: float, matched_by: str) -> dict[str, Any]:
        return {
            "ok": True,
            "project_id": project["id"],
            "name": project["name"],
            "root": project["root"],
            "confidence": confidence,
            "matched_by": matched_by,
            "alternatives": [],
            "error": None,
        }

    def active(self) -> dict[str, Any]:
        state = self.read()
        project = next((item for item in self.projects() if item["id"] == state.get("active_project_id")), None)
        return {"ok": bool(project), "project": project, "active_project_id": state.get("active_project_id")}

    def note_file(self, project_id: str, relative_path: str) -> None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            recent = [
                item
                for item in state.get("recent_files", [])
                if not (item.get("project_id") == project_id and item.get("path") == relative_path)
            ]
            recent.append({"project_id": project_id, "path": relative_path, "used_at": utc_now()})
            state["recent_files"] = recent[-12:]
            state["last_tool_project_id"] = project_id
            state["active_project_id"] = project_id
            state["updated_at"] = utc_now()
            self._write_unlocked(state)

    def _index_path(self, project_id: str) -> Path:
        return self.index_dir / f"{project_id}.json"

    @staticmethod
    def _ignored(path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        parts = [part.casefold() for part in relative.parts]
        if any(
            part in IGNORED_DIRECTORIES
            or part.startswith(".venv-")
            or part.startswith("pytest-")
            for part in parts[:-1]
        ):
            return True
        name = path.name.casefold()
        return name in SENSITIVE_NAMES or path.suffix.casefold() in IGNORED_SUFFIXES

    @staticmethod
    def _model_directory(path: Path) -> bool:
        try:
            names = {item.name.casefold() for item in path.iterdir() if item.is_file()}
        except OSError:
            return False
        has_weights = any(
            any(name.endswith(suffix) for suffix in (".safetensors", ".gguf", ".onnx", ".pt", ".pth"))
            for name in names
        )
        return has_weights and bool({"config.json", "tokenizer.json"}.intersection(names))

    def refresh_index(self, project_id: str) -> dict[str, Any]:
        project = next((item for item in self.projects() if item["id"] == project_id), None)
        if project is None:
            return {"ok": False, "error": "project_not_found", "project_id": project_id}
        root = self.policy.resolve(str(project["root"]))
        previous: dict[str, dict[str, Any]] = {}
        index_path = self._index_path(project_id)
        try:
            value = json.loads(index_path.read_text(encoding="utf-8"))
            previous = {
                item["path"]: item
                for item in value.get("files", [])
                if isinstance(item, dict) and item.get("path")
            }
        except (OSError, json.JSONDecodeError, ValueError):
            previous = {}
        files: list[dict[str, Any]] = []
        for current, directories, names in os.walk(str(root), followlinks=False):
            current_path = Path(current)
            safe_directories = []
            for name in directories:
                child = current_path / name
                if self._ignored(child, root) or child.is_symlink() or self._model_directory(child):
                    continue
                try:
                    self.policy.resolve(str(child))
                except (OSError, AccessDenied):
                    continue
                safe_directories.append(name)
            directories[:] = safe_directories
            for name in names:
                path = current_path / name
                if self._ignored(path, root) or path.is_symlink():
                    continue
                try:
                    resolved = self.policy.resolve(str(path))
                    stat = resolved.stat()
                except (OSError, AccessDenied):
                    continue
                relative = resolved.relative_to(root).as_posix()
                old = previous.get(relative)
                if old and old.get("size") == stat.st_size and old.get("modified_ns") == stat.st_mtime_ns:
                    files.append(old)
                    continue
                files.append(
                    {
                        "path": relative,
                        "extension": resolved.suffix.casefold(),
                        "size": stat.st_size,
                        "modified_ns": stat.st_mtime_ns,
                    }
                )
        payload = {
            "version": 1,
            "project_id": project_id,
            "root": str(root),
            "markers": project.get("markers", []),
            "updated_at": utc_now(),
            "files": sorted(files, key=lambda item: item["path"].casefold()),
        }
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")
        os.replace(temporary, index_path)
        return {"ok": True, "project_id": project_id, "files": len(files), "path": str(index_path)}

    def _load_index(self, project_id: str) -> dict[str, Any]:
        path = self._index_path(project_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("project_id") != project_id or not isinstance(value.get("files"), list):
                raise ValueError("indice invalido")
            return value
        except (OSError, json.JSONDecodeError, ValueError):
            self.refresh_index(project_id)
            return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _query_tokens(query: str) -> tuple[set[str], set[str]]:
        normalized = normalize_name(query)
        words = {
            word
            for word in re.findall(r"[a-z0-9_+-]+", normalized.replace(".", " "))
            if len(word) > 1 and word not in STOP_WORDS
        }
        suffix = Path(normalized).suffix.casefold().lstrip(".")
        if suffix:
            words.discard(suffix)
        expanded = set(words)
        for word in words:
            expanded.update(DESCRIPTION_TERMS.get(word, ()))
        return words, expanded

    def find_files(
        self,
        *,
        project_id: str,
        query: str,
        file_types: list[str] | None = None,
        max_results: int = 20,
    ) -> dict[str, Any]:
        project = next((item for item in self.projects() if item["id"] == project_id), None)
        if project is None:
            return {"ok": False, "error": "project_not_found", "project_id": project_id}
        root = self.policy.resolve(str(project["root"]))
        self.refresh_index(project_id)
        index = self._load_index(project_id)
        requested_types = {
            item.casefold() if item.startswith(".") else f".{item.casefold()}"
            for item in (file_types or [])
        }
        raw_query = normalize_name(query)
        words, expanded = self._query_tokens(query)
        basename_query = Path(raw_query).name.casefold()
        query_extension = Path(basename_query).suffix.casefold()
        scored: list[tuple[float, dict[str, Any], list[str]]] = []
        for item in index["files"]:
            if requested_types and item.get("extension") not in requested_types:
                continue
            relative = str(item["path"])
            path_text = normalize_name(relative)
            basename = Path(relative).name.casefold()
            reasons: list[str] = []
            score = 0.0
            if basename_query and basename == basename_query:
                score = 1.0
                reasons.append("nome exato")
            elif basename_query and basename_query in basename:
                score = max(score, 0.86)
                reasons.append("nome parcial")
            direct_hits = {word for word in words if word in path_text}
            semantic_hits = {word for word in expanded.difference(words) if word in path_text}
            if direct_hits:
                score = max(score, min(0.92, 0.52 + 0.12 * len(direct_hits)))
                reasons.append("termos no caminho: " + ", ".join(sorted(direct_hits)[:4]))
            if semantic_hits:
                score = max(score, min(0.82, 0.46 + 0.08 * len(semantic_hits)))
                reasons.append("descrição relacionada: " + ", ".join(sorted(semantic_hits)[:4]))
            if raw_query.startswith(".") and item.get("extension") == raw_query:
                score = max(score, 0.75)
                reasons.append("extensão")
            elif query_extension and item.get("extension") == query_extension and score > 0:
                score = min(0.95, score + 0.04)
                reasons.append("mesma extensão")
            if "daniel" in words:
                daniel_hints = {
                    ".env.example": (0.98, "configuração declarativa da voz"),
                    "tern/orchestrator/config.py": (0.96, "valor padrão em runtime"),
                    "tern/orchestrator/voice/windows_speech.py": (0.94, "provider Microsoft Daniel"),
                    "docs/voice.md": (0.90, "documentação da voz"),
                }
                hint = daniel_hints.get(relative.casefold())
                if hint:
                    score = max(score, hint[0])
                    reasons.append(hint[1])
            if score > 0:
                scored.append((score, item, reasons))

        if not scored and words:
            text_suffixes = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".ini", ".cfg"}
            inspected = 0
            for item in index["files"]:
                if inspected >= 120 or item.get("extension") not in text_suffixes or int(item.get("size") or 0) > 262144:
                    continue
                inspected += 1
                path = self.policy.resolve(str(root / str(item["path"])))
                try:
                    text = normalize_name(path.read_text(encoding="utf-8", errors="ignore")[:65536])
                except OSError:
                    continue
                hits = {word for word in expanded if word in text}
                if hits:
                    scored.append((min(0.72, 0.42 + 0.04 * len(hits)), item, ["texto limitado relacionado: " + ", ".join(sorted(hits)[:4])]))

        scored.sort(key=lambda value: (-value[0], len(str(value[1]["path"])), str(value[1]["path"]).casefold()))
        if scored:
            relevance_floor = max(0.50, scored[0][0] - 0.35)
            scored = [item for item in scored if item[0] >= relevance_floor]
        results = []
        for score, item, reasons in scored[:max_results]:
            relative = str(item["path"])
            results.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": item["size"],
                    "reason": "; ".join(reasons),
                    "confidence": round(score, 2),
                }
            )
        if results:
            self.note_file(project_id, results[0]["path"])
        return {
            "ok": True,
            "project_id": project_id,
            "project": project["name"],
            "root": str(root),
            "query": query,
            "results": results,
            "count": len(results),
            "exact_match": any(Path(item["path"]).name.casefold() == basename_query for item in results),
            "ambiguous": len(results) > 1 and results[0]["confidence"] == results[1]["confidence"],
            "index_updated_at": index.get("updated_at"),
        }

    def context(self) -> dict[str, Any]:
        state = self.read()
        projects = self.projects()
        active = next((item for item in projects if item["id"] == state.get("active_project_id")), None)
        shared = self._shared_project()
        recent = [
            item["path"]
            for item in reversed(state.get("recent_files", []))
            if not active or item.get("project_id") == active.get("id")
        ][:3]
        running = 0
        jobs = getattr(self.codex, "jobs", None)
        if jobs is not None:
            try:
                running = sum(1 for item in jobs.list() if item.get("status") in {"queued", "starting", "running", "steering", "cancelling", "disconnected", "reconnecting"})
            except Exception:
                running = 0
        return {
            "active_project": active,
            "codex_thread_project": str(shared) if shared else None,
            "allowed_roots": len(self.policy.roots),
            "recent_files": recent,
            "running_codex_jobs": running,
        }

    def context_text(self) -> str:
        value = self.context()
        active = value.get("active_project") or {}
        aliases = ", ".join(active.get("aliases", [])[:6]) or "-"
        recent = ", ".join(value.get("recent_files") or []) or "-"
        return (
            f"Active project: {active.get('name') or '-'}\n"
            f"Project root: {active.get('root') or '-'}\n"
            f"Known aliases: {aliases}\n"
            f"Codex thread project: {value.get('codex_thread_project') or '-'}\n"
            f"Allowed roots: {value.get('allowed_roots')}\n"
            f"Recent files: {recent}\n"
            f"Running Codex jobs: {value.get('running_codex_jobs')}"
        )
