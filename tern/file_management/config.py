from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


class ConfigurationError(ValueError):
    """Raised when a file-management policy is unsafe or malformed."""


DEFAULT_CATEGORIES = {
    "documents": [
        ".doc",
        ".docx",
        ".epub",
        ".md",
        ".odt",
        ".pdf",
        ".rtf",
        ".txt",
    ],
    "spreadsheets": [".csv", ".ods", ".tsv", ".xls", ".xlsx"],
    "images": [".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".svg", ".webp"],
    "audio": [".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"],
    "video": [".avi", ".mkv", ".mov", ".mp4", ".webm"],
    "archives": [".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".zip"],
    "code": [
        ".c",
        ".cpp",
        ".css",
        ".go",
        ".h",
        ".html",
        ".java",
        ".js",
        ".json",
        ".py",
        ".rs",
        ".sh",
        ".toml",
        ".ts",
        ".xml",
        ".yaml",
        ".yml",
    ],
    "installers": [".appx", ".deb", ".dmg", ".exe", ".msi", ".pkg", ".rpm"],
}

LAYOUTS = {"category", "category/year", "category/year/month"}


def _expand_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("paths must be non-empty strings")
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    return Path(expanded).resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_filesystem_root(path: Path) -> bool:
    return path == Path(path.anchor)


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be a list")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be true or false")
    return value


@dataclass(frozen=True)
class OrganizationPolicy:
    root: Path
    recursive: bool = False
    minimum_age_seconds: int = 60
    layout: str = "category/year/month"


@dataclass(frozen=True)
class TransferPolicy:
    source: Path
    destination: Path
    overwrite_newer_destination: bool = False


@dataclass(frozen=True)
class NotificationPolicy:
    console: bool = True
    jsonl: bool = True


@dataclass(frozen=True)
class ManagerConfig:
    allowed_roots: Tuple[Path, ...]
    state_dir: Path
    report_dir: Path
    organizations: Tuple[OrganizationPolicy, ...]
    backups: Tuple[TransferPolicy, ...]
    synchronizations: Tuple[TransferPolicy, ...]
    extension_categories: Mapping[str, str]
    category_names: Tuple[str, ...]
    exclude_patterns: Tuple[str, ...]
    interval_seconds: int
    hash_max_bytes: int
    notifications: NotificationPolicy

    def category_for(self, path: Path) -> str:
        return self.extension_categories.get(path.suffix.casefold(), "other")

    def assert_allowed(self, path: Path, *, label: str = "path") -> Path:
        resolved = path.resolve(strict=False)
        if not any(_is_within(resolved, root) for root in self.allowed_roots):
            raise ConfigurationError(f"{label} is outside allowed_roots: {resolved}")
        return resolved


def _parse_categories(raw: Any) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    values = DEFAULT_CATEGORIES if raw is None else _object(raw, "categories")
    extension_categories: Dict[str, str] = {}
    names = []
    for category, extensions in values.items():
        if not isinstance(category, str) or not category.strip():
            raise ConfigurationError("category names must be non-empty strings")
        normalized_category = category.strip().casefold().replace(" ", "-")
        if normalized_category in names or normalized_category == "other":
            raise ConfigurationError(f"duplicate or reserved category: {normalized_category}")
        names.append(normalized_category)
        for extension in _list(extensions, f"categories.{category}"):
            if not isinstance(extension, str) or not extension.strip():
                raise ConfigurationError(f"invalid extension in category {category}")
            normalized = extension.strip().casefold()
            if not normalized.startswith("."):
                normalized = "." + normalized
            previous = extension_categories.get(normalized)
            if previous is not None:
                raise ConfigurationError(
                    f"extension {normalized} belongs to both {previous} and {normalized_category}"
                )
            extension_categories[normalized] = normalized_category
    names.append("other")
    return extension_categories, tuple(names)


def _parse_organizations(raw: Any) -> Tuple[OrganizationPolicy, ...]:
    policies = []
    for index, item in enumerate(_list(raw or [], "organization")):
        value = _object(item, f"organization[{index}]")
        root = _expand_path(value.get("root"))
        if _is_filesystem_root(root):
            raise ConfigurationError("organizing a filesystem root is forbidden")
        minimum_age = int(value.get("minimum_age_seconds", 60))
        if minimum_age < 0:
            raise ConfigurationError("minimum_age_seconds cannot be negative")
        layout = str(value.get("layout", "category/year/month"))
        if layout not in LAYOUTS:
            raise ConfigurationError(f"unsupported organization layout: {layout}")
        policies.append(
            OrganizationPolicy(
                root=root,
                recursive=_boolean(
                    value.get("recursive", False),
                    f"organization[{index}].recursive",
                ),
                minimum_age_seconds=minimum_age,
                layout=layout,
            )
        )
    return tuple(policies)


def _parse_transfers(raw: Any, name: str) -> Tuple[TransferPolicy, ...]:
    policies = []
    for index, item in enumerate(_list(raw or [], name)):
        value = _object(item, f"{name}[{index}]")
        source = _expand_path(value.get("source"))
        destination = _expand_path(value.get("destination"))
        if _is_filesystem_root(source) or _is_filesystem_root(destination):
            raise ConfigurationError(f"filesystem roots are forbidden in {name}")
        if _is_within(destination, source) or _is_within(source, destination):
            raise ConfigurationError(
                f"overlapping source and destination in {name}: {source} <-> {destination}"
            )
        policies.append(
            TransferPolicy(
                source=source,
                destination=destination,
                overwrite_newer_destination=_boolean(
                    value.get("overwrite_newer_destination", False),
                    f"{name}[{index}].overwrite_newer_destination",
                ),
            )
        )
    return tuple(policies)


def _validate_allowed(config: ManagerConfig) -> None:
    if not config.allowed_roots:
        raise ConfigurationError("allowed_roots cannot be empty")
    for root in config.allowed_roots:
        if _is_filesystem_root(root):
            raise ConfigurationError("filesystem roots cannot be allowlisted")
    controlled_paths: Iterable[Tuple[str, Path]] = (
        [("organization root", item.root) for item in config.organizations]
        + [("backup source", item.source) for item in config.backups]
        + [("backup destination", item.destination) for item in config.backups]
        + [("sync source", item.source) for item in config.synchronizations]
        + [("sync destination", item.destination) for item in config.synchronizations]
    )
    for label, path in controlled_paths:
        config.assert_allowed(path, label=label)


def load_config(path: Path) -> ManagerConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc
    root = _object(raw, "configuration")
    allowed_roots = tuple(
        _expand_path(item) for item in _list(root.get("allowed_roots"), "allowed_roots")
    )
    state_dir = _expand_path(root.get("state_dir", "~/.jarvis-file-manager"))
    report_dir = _expand_path(root.get("report_dir", str(state_dir / "reports")))
    if _is_filesystem_root(state_dir) or _is_filesystem_root(report_dir):
        raise ConfigurationError("state_dir and report_dir cannot be filesystem roots")
    extension_categories, category_names = _parse_categories(root.get("categories"))
    notifications_raw = _object(root.get("notifications", {}), "notifications")
    interval_seconds = int(root.get("interval_seconds", 900))
    hash_max_bytes = int(root.get("hash_max_bytes", 104_857_600))
    if interval_seconds < 10:
        raise ConfigurationError("interval_seconds must be at least 10")
    if hash_max_bytes < 0:
        raise ConfigurationError("hash_max_bytes cannot be negative")
    config = ManagerConfig(
        allowed_roots=allowed_roots,
        state_dir=state_dir,
        report_dir=report_dir,
        organizations=_parse_organizations(root.get("organization")),
        backups=_parse_transfers(root.get("backups"), "backups"),
        synchronizations=_parse_transfers(
            root.get("synchronizations"), "synchronizations"
        ),
        extension_categories=extension_categories,
        category_names=category_names,
        exclude_patterns=tuple(
            str(item) for item in _list(root.get("exclude_patterns", []), "exclude_patterns")
        ),
        interval_seconds=interval_seconds,
        hash_max_bytes=hash_max_bytes,
        notifications=NotificationPolicy(
            console=_boolean(
                notifications_raw.get("console", True),
                "notifications.console",
            ),
            jsonl=_boolean(
                notifications_raw.get("jsonl", True),
                "notifications.jsonl",
            ),
        ),
    )
    _validate_allowed(config)
    return config
