"""Bounded, read-only discovery of named projects across configured roots."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable



def normalize_name(value: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9._+ -]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


PROJECT_MARKERS: dict[str, int] = {
    ".git": 5,
    "package.json": 4,
    "pyproject.toml": 4,
    "requirements.txt": 3,
    "Pipfile": 3,
    "poetry.lock": 3,
    "Cargo.toml": 4,
    "go.mod": 4,
    "pom.xml": 4,
    "build.gradle": 3,
    "build.gradle.kts": 3,
    "*.sln": 4,
    "*.csproj": 4,
    "firebase.json": 3,
    "docker-compose.yml": 3,
    "compose.yml": 3,
    "Makefile": 2,
    "CMakeLists.txt": 3,
}

DEFAULT_EXCLUDES = frozenset(
    {
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "system volume information",
        "$recycle.bin",
        "recovery",
        "appdata",
        ".ssh",
        ".aws",
        ".azure",
        ".gnupg",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        "target",
        ".cache",
        "cache",
        "coverage",
        ".next",
        ".nuxt",
        "out",
        "tmp",
        "temp",
        "__pycache__",
    }
)


def default_discovery_roots() -> tuple[Path, ...]:
    home = Path.home()
    values = [
        home / "Desktop",
        home / "Documents",
        home / "Projects",
        home / "source",
        home / "repos",
        home / "dev",
        Path("C:/Projects"),
        Path("C:/Projetos"),
        Path("C:/Dev"),
        Path("C:/Repos"),
        Path("D:/"),
        Path("D:/Projects"),
        Path("D:/Projetos"),
        Path("D:/Dev"),
        Path("D:/Repos"),
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = os.path.normcase(str(value.resolve(strict=False))).casefold()
        if key not in seen and value.exists():
            seen.add(key)
            result.append(value.resolve(strict=False))
    return tuple(result)


@dataclass(frozen=True)
class DiscoveryPolicy:
    roots: tuple[Path, ...]
    excludes: frozenset[str] = DEFAULT_EXCLUDES
    max_depth: int = 3
    drive_max_depth: int = 2
    max_directories: int = 4000
    max_candidates: int = 20
    max_elapsed_ms: int = 1500

    @classmethod
    def from_values(
        cls,
        roots: Iterable[str | Path] | None = None,
        excludes: Iterable[str] | None = None,
        **overrides: int,
    ) -> "DiscoveryPolicy":
        selected = tuple(Path(item).expanduser().resolve(strict=False) for item in (roots or default_discovery_roots()))
        excluded = frozenset(
            str(item).strip().casefold()
            for item in (excludes or DEFAULT_EXCLUDES)
            if str(item).strip()
        )
        return cls(roots=selected, excludes=excluded, **overrides)


@dataclass(frozen=True)
class ProjectCandidate:
    name: str
    path: str
    matched_reference: str
    match_score: float
    match_reasons: tuple[str, ...]
    markers: tuple[str, ...]
    discovery_root: str
    identity_confidence: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "matched_reference": self.matched_reference,
            "match_score": self.match_score,
            "match_reasons": list(self.match_reasons),
            "markers": list(self.markers),
            "discovery_root": self.discovery_root,
            "identity_confidence": self.identity_confidence,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    status: str
    reference: str
    candidates: tuple[ProjectCandidate, ...] = ()
    roots_checked: tuple[str, ...] = ()
    directories_checked: int = 0
    elapsed_ms: float = 0.0
    budget_reached: bool = False
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        selected = self.candidates[0] if self.status == "RESOLVED" and self.candidates else None
        return {
            "ok": self.status == "RESOLVED",
            "status": self.status,
            "reference": self.reference,
            "project": selected.as_dict() if selected else None,
            "candidates": [item.as_dict() for item in self.candidates],
            "roots_checked": list(self.roots_checked),
            "directories_checked": self.directories_checked,
            "elapsed_ms": self.elapsed_ms,
            "budget_reached": self.budget_reached,
            "errors": list(self.errors),
        }


class ProjectDiscovery:
    """Discovery plane. It never uses project mutation tools or executes code."""

    def __init__(self, policy: DiscoveryPolicy):
        self.policy = policy

    @staticmethod
    def _drive_root(path: Path) -> bool:
        return path.parent == path

    def _excluded(self, path: Path, root: Path | None = None) -> bool:
        if root is None:
            parts = (path.name,)
        else:
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                parts = (path.name,)
        return any(part.casefold() in self.policy.excludes for part in parts)

    @staticmethod
    def _markers(path: Path) -> tuple[tuple[str, ...], int]:
        markers: list[str] = []
        score = 0
        try:
            names = {entry.name.casefold(): entry.name for entry in os.scandir(path)}
        except (OSError, PermissionError):
            return (), 0
        for marker, points in PROJECT_MARKERS.items():
            if marker.startswith("*."):
                suffix = marker[1:].casefold()
                found = any(name.endswith(suffix) for name in names)
            else:
                found = marker.casefold() in names
            if found:
                markers.append(marker)
                score += points
        return tuple(sorted(markers)), score

    @staticmethod
    def _identity_score(path: Path, reference: str) -> tuple[float, list[str]]:
        normalized_ref = normalize_name(reference)
        normalized_name = normalize_name(path.name)
        reference_terms = [term for term in normalized_ref.split() if len(term) >= 3]
        if normalized_name == normalized_ref or normalized_name in reference_terms:
            return 1.0, ["DIRECTORY_NAME_EXACT_MATCH"]
        if normalized_ref and normalized_ref in normalized_name:
            return 0.65, ["DIRECTORY_NAME_CONTAINS_MATCH"]
        return 0.0, []

    @staticmethod
    def _manifest_identity(path: Path, reference: str) -> bool:
        reference_name = normalize_name(reference)
        reference_names = {reference_name, *[term for term in reference_name.split() if len(term) >= 3]}
        for filename in ("package.json", "pyproject.toml"):
            candidate = path / filename
            try:
                if filename == "package.json":
                    value = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
                    names = [value.get("name")] if isinstance(value, dict) else []
                else:
                    text = candidate.read_text(encoding="utf-8", errors="replace")[:12000]
                    names = [line.split("=", 1)[1].strip().strip('"\'') for line in text.splitlines() if line.strip().startswith("name") and "=" in line]
                if any(normalize_name(str(name)) in reference_names for name in names if name):
                    return True
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return False

    def discover(self, reference: str) -> DiscoveryResult:
        started = time.perf_counter()
        clean_reference = str(reference or "").strip()
        if not clean_reference:
            return DiscoveryResult("NOT_FOUND", clean_reference)
        candidates: list[ProjectCandidate] = []
        seen: set[str] = set()
        roots_checked: list[str] = []
        errors: list[str] = []
        directories_checked = 0
        budget_reached = False
        for configured in self.policy.roots:
            if budget_reached:
                break
            root = configured.resolve(strict=False)
            if not root.exists() or root.name.casefold() in self.policy.excludes:
                continue
            roots_checked.append(str(root))
            max_depth = self.policy.drive_max_depth if self._drive_root(root) else self.policy.max_depth
            queue: list[tuple[Path, int]] = [(root, 0)]
            while queue:
                if directories_checked >= self.policy.max_directories or (time.perf_counter() - started) * 1000 >= self.policy.max_elapsed_ms:
                    budget_reached = True
                    break
                current, depth = queue.pop(0)
                if self._excluded(current, root) or current.is_symlink():
                    continue
                directories_checked += 1
                markers, marker_score = self._markers(current)
                identity_score, reasons = self._identity_score(current, clean_reference)
                if identity_score and markers:
                    if self._manifest_identity(current, clean_reference):
                        identity_score = min(1.0, identity_score + 0.05)
                        reasons.append("MANIFEST_NAME_MATCH")
                    reasons.append("PROJECT_MARKER_FOUND")
                    if ".git" in markers:
                        reasons.append("GIT_REPOSITORY")
                    key = str(current.resolve(strict=False)).casefold()
                    if key not in seen:
                        seen.add(key)
                        score = round(min(0.999, identity_score * 0.75 + min(marker_score / 20, 0.24)), 3)
                        candidates.append(ProjectCandidate(
                            name=current.name,
                            path=str(current.resolve(strict=False)),
                            matched_reference=clean_reference,
                            match_score=score,
                            match_reasons=tuple(dict.fromkeys(reasons)),
                            markers=markers,
                            discovery_root=str(root),
                            identity_confidence="HIGH" if score >= 0.85 else "MEDIUM",
                        ))
                if len(candidates) >= self.policy.max_candidates:
                    budget_reached = True
                    break
                if depth >= max_depth:
                    continue
                try:
                    children = sorted((entry for entry in os.scandir(current) if entry.is_dir(follow_symlinks=False)), key=lambda item: item.name.casefold())
                except (OSError, PermissionError) as exc:
                    errors.append(f"{current}:{type(exc).__name__}")
                    continue
                queue.extend((Path(entry.path), depth + 1) for entry in children if entry.name.casefold() not in self.policy.excludes)
        candidates.sort(key=lambda item: (-item.match_score, item.path.casefold()))
        high = [item for item in candidates if item.match_score >= 0.85]
        status = "RESOLVED" if len(high) == 1 else "AMBIGUOUS" if len(high) > 1 or len(candidates) > 1 else "NOT_FOUND"
        return DiscoveryResult(
            status=status,
            reference=clean_reference,
            candidates=tuple(candidates),
            roots_checked=tuple(roots_checked),
            directories_checked=directories_checked,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            budget_reached=budget_reached,
            errors=tuple(errors[:20]),
        )

    def inspect_path(self, path: str | Path, reference: str | None = None) -> ProjectCandidate | None:
        """Validate one user-provided path without granting its parent/root."""
        candidate = Path(path).expanduser().resolve(strict=False)
        if not candidate.is_dir() or candidate.is_symlink():
            return None
        if not any(candidate == root or root in candidate.parents for root in self.policy.roots):
            return None
        markers, marker_score = self._markers(candidate)
        if not markers:
            return None
        ref = str(reference or candidate.name)
        identity_score, reasons = self._identity_score(candidate, ref)
        if not identity_score:
            identity_score = 0.8
            reasons = ["USER_PATH_EXPLICIT"]
        reasons.append("PROJECT_MARKER_FOUND")
        return ProjectCandidate(
            name=candidate.name,
            path=str(candidate),
            matched_reference=ref,
            match_score=round(min(0.999, identity_score * 0.75 + min(marker_score / 20, 0.24)), 3),
            match_reasons=tuple(dict.fromkeys(reasons)),
            markers=markers,
            discovery_root=str(next(root for root in self.policy.roots if candidate == root or root in candidate.parents)),
            identity_confidence="HIGH" if identity_score >= 0.85 else "MEDIUM",
        )
