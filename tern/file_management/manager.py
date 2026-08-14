from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .config import (
    ConfigurationError,
    ManagerConfig,
    OrganizationPolicy,
    TransferPolicy,
)
from .storage import (
    ActionJournal,
    FileAction,
    FileRecord,
    MetadataStore,
    Reporter,
    utc_now,
)


class FileManager:
    """Non-destructive file organizer, inventory, backup, and synchronizer."""

    def __init__(self, config: ManagerConfig):
        self.config = config
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = MetadataStore(self.config.state_dir / "metadata.sqlite3")
        self.journal = ActionJournal(self.config.state_dir / "actions.jsonl")
        self.reporter = Reporter(
            self.config.report_dir,
            self.config.state_dir / "notifications.jsonl",
        )

    def validate_runtime(self) -> Dict[str, Any]:
        missing_sources = []
        invalid_sources = []
        sources = (
            [item.root for item in self.config.organizations]
            + [item.source for item in self.config.backups]
            + [item.source for item in self.config.synchronizations]
        )
        for source in self._unique_paths(sources):
            if not source.exists():
                missing_sources.append(str(source))
            elif not source.is_dir():
                invalid_sources.append(str(source))
        return {
            "ok": not missing_sources and not invalid_sources,
            "allowed_roots": [str(item) for item in self.config.allowed_roots],
            "state_dir": str(self.config.state_dir),
            "report_dir": str(self.config.report_dir),
            "missing_sources": missing_sources,
            "invalid_sources": invalid_sources,
            "organization_policies": len(self.config.organizations),
            "backup_policies": len(self.config.backups),
            "synchronization_policies": len(self.config.synchronizations),
        }

    def organize(self, *, apply: bool = False) -> Tuple[List[FileAction], List[str]]:
        actions: List[FileAction] = []
        errors: List[str] = []
        now = time.time()
        for policy in self.config.organizations:
            if not self._valid_source(policy.root, "organization", errors):
                continue
            for source in self._iter_files(policy.root, recursive=policy.recursive):
                try:
                    relative = source.relative_to(policy.root)
                    if self._excluded(source, relative):
                        continue
                    if policy.recursive and relative.parts[0].casefold() in self.config.category_names:
                        continue
                    stat = source.stat()
                    if now - stat.st_mtime < policy.minimum_age_seconds:
                        continue
                    category = self.config.category_for(source)
                    destination_dir = self._organization_directory(
                        policy, category, stat.st_mtime
                    )
                    destination = self._collision_free(destination_dir / source.name)
                    self.config.assert_allowed(destination, label="organization destination")
                    action = FileAction(
                        operation="organize",
                        source=str(source),
                        destination=str(destination),
                        status="planned" if not apply else "applied",
                        bytes=stat.st_size,
                    )
                    if apply:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        self._move_without_overwrite(source, destination)
                    self._record(action, actions)
                except (OSError, ConfigurationError) as exc:
                    message = f"organize {source}: {exc}"
                    errors.append(message)
                    self._record(
                        FileAction(
                            operation="organize",
                            source=str(source),
                            destination=None,
                            status="failed",
                            reason=str(exc),
                        ),
                        actions,
                    )
        return actions, errors

    def scan_metadata(self) -> Tuple[Dict[str, int], List[str]]:
        started_at = utc_now()
        errors: List[str] = []
        records: List[FileRecord] = []
        roots = self._metadata_roots()
        for root in roots:
            if not self._valid_source(root, "metadata", errors):
                continue
            for path in self._iter_files(root, recursive=True):
                try:
                    relative = path.relative_to(root)
                    if self._excluded(path, relative):
                        continue
                    stat = path.stat()
                    path_text = str(path)
                    digest = self.metadata.cached_hash(
                        path_text, stat.st_size, stat.st_mtime_ns
                    )
                    if digest is None and stat.st_size <= self.config.hash_max_bytes:
                        digest = self._sha256(path)
                    records.append(
                        FileRecord(
                            path=path_text,
                            root=str(root),
                            size=stat.st_size,
                            mtime_ns=stat.st_mtime_ns,
                            extension=path.suffix.casefold(),
                            category=self.config.category_for(path),
                            sha256=digest,
                        )
                    )
                except OSError as exc:
                    errors.append(f"metadata {path}: {exc}")
        scanned_roots = [item for item in roots if item.is_dir()]
        result = self.metadata.update_scan(
            records,
            scanned_roots,
            started_at=started_at,
        )
        return result, errors

    def backup(self, *, apply: bool = False) -> Tuple[List[FileAction], List[str]]:
        actions: List[FileAction] = []
        errors: List[str] = []
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for policy in self.config.backups:
            if not self._valid_source(policy.source, "backup", errors):
                continue
            for source in self._iter_files(policy.source, recursive=True):
                try:
                    relative = source.relative_to(policy.source)
                    if self._excluded(source, relative):
                        continue
                    destination = policy.destination / relative
                    self.config.assert_allowed(destination, label="backup destination")
                    if destination.is_symlink():
                        raise OSError(f"refusing symlink destination: {destination}")
                    if destination.exists() and not self._different(source, destination):
                        continue
                    source_size = source.stat().st_size
                    if destination.exists():
                        version_path = policy.destination / ".versions" / version / relative
                        self.config.assert_allowed(version_path, label="backup version")
                        version_action = FileAction(
                            operation="backup-version",
                            source=str(destination),
                            destination=str(version_path),
                            status="planned" if not apply else "applied",
                            bytes=destination.stat().st_size,
                        )
                        if apply:
                            self._atomic_copy(destination, version_path)
                        self._record(version_action, actions)
                    action = FileAction(
                        operation="backup",
                        source=str(source),
                        destination=str(destination),
                        status="planned" if not apply else "applied",
                        bytes=source_size,
                    )
                    if apply:
                        self._atomic_copy(source, destination)
                    self._record(action, actions)
                except (OSError, ConfigurationError) as exc:
                    errors.append(f"backup {source}: {exc}")
                    self._record(
                        FileAction(
                            operation="backup",
                            source=str(source),
                            destination=None,
                            status="failed",
                            reason=str(exc),
                        ),
                        actions,
                    )
        return actions, errors

    def synchronize(self, *, apply: bool = False) -> Tuple[List[FileAction], List[str]]:
        actions: List[FileAction] = []
        errors: List[str] = []
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for policy in self.config.synchronizations:
            if not self._valid_source(policy.source, "sync", errors):
                continue
            for source in self._iter_files(policy.source, recursive=True):
                try:
                    relative = source.relative_to(policy.source)
                    if self._excluded(source, relative):
                        continue
                    destination = policy.destination / relative
                    self.config.assert_allowed(destination, label="sync destination")
                    if destination.is_symlink():
                        raise OSError(f"refusing symlink destination: {destination}")
                    if destination.exists() and not self._different(source, destination):
                        continue
                    if (
                        destination.exists()
                        and destination.stat().st_mtime_ns > source.stat().st_mtime_ns
                        and not policy.overwrite_newer_destination
                    ):
                        self._record(
                            FileAction(
                                operation="sync",
                                source=str(source),
                                destination=str(destination),
                                status="skipped",
                                reason="destination is newer",
                            ),
                            actions,
                        )
                        continue
                    if destination.exists():
                        version_path = policy.destination / ".versions" / version / relative
                        self.config.assert_allowed(version_path, label="sync version")
                        version_action = FileAction(
                            operation="sync-version",
                            source=str(destination),
                            destination=str(version_path),
                            status="planned" if not apply else "applied",
                            bytes=destination.stat().st_size,
                        )
                        if apply:
                            self._atomic_copy(destination, version_path)
                        self._record(version_action, actions)
                    action = FileAction(
                        operation="sync",
                        source=str(source),
                        destination=str(destination),
                        status="planned" if not apply else "applied",
                        bytes=source.stat().st_size,
                    )
                    if apply:
                        self._atomic_copy(source, destination)
                    self._record(action, actions)
                except (OSError, ConfigurationError) as exc:
                    errors.append(f"sync {source}: {exc}")
                    self._record(
                        FileAction(
                            operation="sync",
                            source=str(source),
                            destination=None,
                            status="failed",
                            reason=str(exc),
                        ),
                        actions,
                    )
        return actions, errors

    def run_cycle(self, *, apply: bool = False) -> Dict[str, Any]:
        started_at = utc_now()
        actions: List[FileAction] = []
        errors: List[str] = []

        organize_actions, organize_errors = self.organize(apply=apply)
        actions.extend(organize_actions)
        errors.extend(organize_errors)

        metadata, metadata_errors = self.scan_metadata()
        errors.extend(metadata_errors)

        backup_actions, backup_errors = self.backup(apply=apply)
        actions.extend(backup_actions)
        errors.extend(backup_errors)

        sync_actions, sync_errors = self.synchronize(apply=apply)
        actions.extend(sync_actions)
        errors.extend(sync_errors)

        summary = self._summarize(actions)
        report: Dict[str, Any] = {
            "status": "failed" if errors else "ok",
            "mode": "apply" if apply else "dry-run",
            "started_at": started_at,
            "finished_at": utc_now(),
            "summary": summary,
            "metadata": metadata,
            "errors": errors,
            "actions": [item.as_dict() for item in actions[:2000]],
            "actions_truncated": max(0, len(actions) - 2000),
            "safety": {
                "deletions_enabled": False,
                "symlinks_followed": False,
                "allowed_roots": [str(item) for item in self.config.allowed_roots],
            },
        }
        paths = self.reporter.write(
            report,
            notify_jsonl=self.config.notifications.jsonl,
        )
        report["reports"] = paths
        if self.config.notifications.console:
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "mode": report["mode"],
                        "summary": summary,
                        "report": paths["markdown"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return report

    def _metadata_roots(self) -> List[Path]:
        return self._unique_paths(
            [item.root for item in self.config.organizations]
            + [item.source for item in self.config.backups]
            + [item.source for item in self.config.synchronizations]
        )

    @staticmethod
    def _unique_paths(paths: Sequence[Path]) -> List[Path]:
        result: List[Path] = []
        seen: Set[str] = set()
        for path in paths:
            identity = os.path.normcase(str(path.resolve(strict=False)))
            if identity not in seen:
                seen.add(identity)
                result.append(path.resolve(strict=False))
        return result

    def _valid_source(self, root: Path, operation: str, errors: List[str]) -> bool:
        try:
            self.config.assert_allowed(root, label=f"{operation} source")
            if not root.exists():
                raise FileNotFoundError(root)
            if not root.is_dir():
                raise NotADirectoryError(root)
            if root.is_symlink():
                raise OSError(f"refusing symlink source: {root}")
            return True
        except (OSError, ConfigurationError) as exc:
            errors.append(f"{operation} source {root}: {exc}")
            return False

    def _iter_files(self, root: Path, *, recursive: bool) -> Iterator[Path]:
        if not recursive:
            for item in sorted(root.iterdir(), key=lambda value: value.name.casefold()):
                if item.is_file() and not item.is_symlink():
                    yield item.resolve(strict=True)
            return
        for current, directories, files in os.walk(str(root), followlinks=False):
            current_path = Path(current)
            directories[:] = sorted(
                [
                    name
                    for name in directories
                    if not (current_path / name).is_symlink()
                    and not self._excluded(current_path / name, (current_path / name).relative_to(root))
                ],
                key=str.casefold,
            )
            for name in sorted(files, key=str.casefold):
                item = current_path / name
                if item.is_symlink() or not item.is_file():
                    continue
                yield item.resolve(strict=True)

    def _excluded(self, path: Path, relative: Path) -> bool:
        path_resolved = path.resolve(strict=False)
        for internal in (self.config.state_dir, self.config.report_dir):
            try:
                path_resolved.relative_to(internal.resolve(strict=False))
                return True
            except ValueError:
                pass
        relative_text = relative.as_posix()
        return any(
            fnmatch.fnmatch(path.name, pattern)
            or fnmatch.fnmatch(relative_text, pattern)
            for pattern in self.config.exclude_patterns
        )

    @staticmethod
    def _organization_directory(
        policy: OrganizationPolicy,
        category: str,
        modified_at: float,
    ) -> Path:
        date = datetime.fromtimestamp(modified_at)
        destination = policy.root / category
        if policy.layout in {"category/year", "category/year/month"}:
            destination /= f"{date.year:04d}"
        if policy.layout == "category/year/month":
            destination /= f"{date.month:02d}"
        return destination

    @staticmethod
    def _collision_free(destination: Path) -> Path:
        if not destination.exists():
            return destination
        for index in range(1, 10_001):
            candidate = destination.with_name(
                f"{destination.stem} ({index}){destination.suffix}"
            )
            if not candidate.exists():
                return candidate
        raise OSError(f"too many name collisions for {destination}")

    @staticmethod
    def _move_without_overwrite(source: Path, destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        try:
            os.link(str(source), str(destination), follow_symlinks=False)
        except TypeError:
            os.link(str(source), str(destination))
        except OSError:
            created = False
            try:
                with source.open("rb") as source_handle, destination.open("xb") as target:
                    created = True
                    shutil.copyfileobj(source_handle, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                shutil.copystat(source, destination, follow_symlinks=False)
            except Exception:
                if created and destination.exists():
                    destination.unlink()
                raise
        try:
            source.unlink()
        except OSError:
            # Keeping both copies is safer than risking data loss.
            raise

    def _different(self, source: Path, destination: Path) -> bool:
        source_stat = source.stat()
        destination_stat = destination.stat()
        if source_stat.st_size != destination_stat.st_size:
            return True
        if source_stat.st_mtime_ns != destination_stat.st_mtime_ns:
            return True
        if source_stat.st_size <= self.config.hash_max_bytes:
            return self._sha256(source) != self._sha256(destination)
        return False

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source, temporary, follow_symlinks=False)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _record(self, action: FileAction, actions: List[FileAction]) -> None:
        actions.append(action)
        self.journal.write(action)

    @staticmethod
    def _summarize(actions: Iterable[FileAction]) -> Dict[str, int]:
        result = {
            "planned": 0,
            "applied": 0,
            "skipped": 0,
            "failed": 0,
            "bytes_copied": 0,
        }
        for action in actions:
            if action.status in result:
                result[action.status] += 1
            if action.status == "applied" and action.operation in {
                "backup",
                "backup-version",
                "sync",
                "sync-version",
            }:
                result["bytes_copied"] += action.bytes
        return result
