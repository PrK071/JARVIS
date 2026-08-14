from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class FileRecord:
    path: str
    root: str
    size: int
    mtime_ns: int
    extension: str
    category: str
    sha256: Optional[str]


@dataclass(frozen=True)
class FileAction:
    operation: str
    source: str
    destination: Optional[str]
    status: str
    reason: str = ""
    bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataStore:
    """Incremental SQLite inventory with atomic transactions."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    extension TEXT NOT NULL,
                    category TEXT NOT NULL,
                    sha256 TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    missing INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS files_root_missing
                    ON files(root, missing);
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    roots INTEGER NOT NULL,
                    files_seen INTEGER NOT NULL,
                    files_new INTEGER NOT NULL,
                    files_updated INTEGER NOT NULL,
                    files_missing INTEGER NOT NULL
                );
                """
            )

    def cached_hash(self, path: str, size: int, mtime_ns: int) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sha256 FROM files WHERE path=? AND size=? AND mtime_ns=? AND missing=0",
                (path, size, mtime_ns),
            ).fetchone()
        return row[0] if row and isinstance(row[0], str) else None

    def update_scan(
        self,
        records: Sequence[FileRecord],
        roots: Sequence[Path],
        *,
        started_at: str,
    ) -> Dict[str, int]:
        now = utc_now()
        new = 0
        updated = 0
        unchanged = 0
        root_values = [str(item) for item in roots]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if root_values:
                placeholders = ",".join("?" for _ in root_values)
                connection.execute(
                    f"UPDATE files SET missing=1 WHERE root IN ({placeholders})",
                    root_values,
                )
            for record in records:
                previous = connection.execute(
                    "SELECT size, mtime_ns, extension, category, sha256, missing "
                    "FROM files WHERE path=?",
                    (record.path,),
                ).fetchone()
                values = (
                    record.root,
                    record.size,
                    record.mtime_ns,
                    record.extension,
                    record.category,
                    record.sha256,
                    now,
                    record.path,
                )
                if previous is None:
                    connection.execute(
                        "INSERT INTO files "
                        "(root, size, mtime_ns, extension, category, sha256, "
                        " first_seen_at, last_seen_at, path, missing) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                        (
                            record.root,
                            record.size,
                            record.mtime_ns,
                            record.extension,
                            record.category,
                            record.sha256,
                            now,
                            now,
                            record.path,
                        ),
                    )
                    new += 1
                else:
                    current = (
                        record.size,
                        record.mtime_ns,
                        record.extension,
                        record.category,
                        record.sha256,
                    )
                    changed = current != tuple(previous[:5]) or bool(previous[5])
                    connection.execute(
                        "UPDATE files SET root=?, size=?, mtime_ns=?, extension=?, "
                        "category=?, sha256=?, last_seen_at=?, missing=0 WHERE path=?",
                        values,
                    )
                    if changed:
                        updated += 1
                    else:
                        unchanged += 1
            missing = 0
            if root_values:
                placeholders = ",".join("?" for _ in root_values)
                missing = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM files WHERE root IN ({placeholders}) AND missing=1",
                        root_values,
                    ).fetchone()[0]
                )
            connection.execute(
                "INSERT INTO scans "
                "(started_at, finished_at, roots, files_seen, files_new, "
                " files_updated, files_missing) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (started_at, now, len(root_values), len(records), new, updated, missing),
            )
        return {
            "seen": len(records),
            "new": new,
            "updated": updated,
            "unchanged": unchanged,
            "missing": missing,
        }

    def records(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT path, root, size, mtime_ns, extension, category, sha256, "
                "first_seen_at, last_seen_at, missing FROM files ORDER BY path"
            ).fetchall()
        return [dict(row) for row in rows]


class ActionJournal:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, action: FileAction) -> None:
        record = {"time": utc_now(), **action.as_dict()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class Reporter:
    def __init__(self, report_dir: Path, notification_path: Path):
        self.report_dir = report_dir
        self.notification_path = notification_path

    def write(self, report: Dict[str, Any], *, notify_jsonl: bool) -> Dict[str, str]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        identity = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        identity += f"-{time.time_ns() % 1_000_000_000:09d}Z"
        json_path = self.report_dir / f"file-management-{identity}.json"
        markdown_path = self.report_dir / f"file-management-{identity}.md"
        paths = {"json": str(json_path), "markdown": str(markdown_path)}
        persisted_report = {**report, "reports": paths}
        encoded = json.dumps(persisted_report, ensure_ascii=False, indent=2) + "\n"
        markdown = self._markdown(persisted_report)
        atomic_write_text(json_path, encoded)
        atomic_write_text(markdown_path, markdown)
        atomic_write_text(self.report_dir / "latest.json", encoded)
        atomic_write_text(self.report_dir / "latest.md", markdown)
        if notify_jsonl:
            notification = {
                "time": utc_now(),
                "status": report.get("status"),
                "mode": report.get("mode"),
                "summary": report.get("summary"),
                "report": str(markdown_path),
            }
            self.notification_path.parent.mkdir(parents=True, exist_ok=True)
            with self.notification_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(notification, ensure_ascii=False) + "\n")
        return paths

    @staticmethod
    def _markdown(report: Dict[str, Any]) -> str:
        summary = report.get("summary") or {}
        metadata = report.get("metadata") or {}
        errors = report.get("errors") or []
        lines = [
            "# File management report",
            "",
            f"- Status: `{report.get('status', 'unknown')}`",
            f"- Mode: `{report.get('mode', 'unknown')}`",
            f"- Started: `{report.get('started_at', '-')}`",
            f"- Finished: `{report.get('finished_at', '-')}`",
            f"- Planned actions: {summary.get('planned', 0)}",
            f"- Applied actions: {summary.get('applied', 0)}",
            f"- Skipped actions: {summary.get('skipped', 0)}",
            f"- Failed actions: {summary.get('failed', 0)}",
            f"- Bytes copied: {summary.get('bytes_copied', 0)}",
            "",
            "## Metadata",
            "",
            f"- Seen: {metadata.get('seen', 0)}",
            f"- New: {metadata.get('new', 0)}",
            f"- Updated: {metadata.get('updated', 0)}",
            f"- Missing: {metadata.get('missing', 0)}",
            "",
            "## Errors",
            "",
        ]
        lines.extend(f"- {item}" for item in errors)
        if not errors:
            lines.append("- None")
        return "\n".join(lines) + "\n"
