from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tern.file_management import ConfigurationError, FileManager, load_config


def make_config(tmp_path: Path, **overrides) -> Path:
    inbox = tmp_path / "inbox"
    backup_source = tmp_path / "important"
    backup_destination = tmp_path / "backup"
    sync_source = tmp_path / "sync-source"
    sync_destination = tmp_path / "sync-destination"
    for path in (inbox, backup_source, backup_destination, sync_source, sync_destination):
        path.mkdir(parents=True, exist_ok=True)
    value = {
        "allowed_roots": [str(tmp_path)],
        "state_dir": str(tmp_path / "state"),
        "report_dir": str(tmp_path / "reports"),
        "interval_seconds": 10,
        "hash_max_bytes": 1_000_000,
        "categories": {
            "documents": [".txt", ".pdf"],
            "images": [".jpg", ".png"],
        },
        "exclude_patterns": ["*.tmp", ".*"],
        "organization": [
            {
                "root": str(inbox),
                "recursive": True,
                "minimum_age_seconds": 0,
                "layout": "category/year/month",
            }
        ],
        "backups": [
            {
                "source": str(backup_source),
                "destination": str(backup_destination),
            }
        ],
        "synchronizations": [
            {
                "source": str(sync_source),
                "destination": str(sync_destination),
                "overwrite_newer_destination": False,
            }
        ],
        "notifications": {"console": False, "jsonl": True},
    }
    value.update(overrides)
    path = tmp_path / "file-manager.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_configuration_rejects_overlapping_transfer_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = make_config(
        tmp_path,
        backups=[
            {
                "source": str(source),
                "destination": str(source / "nested-backup"),
            }
        ],
    )
    with pytest.raises(ConfigurationError, match="overlapping"):
        load_config(config)


def test_organization_is_dry_run_by_default_and_applies_without_overwrite(tmp_path):
    config = load_config(make_config(tmp_path))
    source = tmp_path / "inbox" / "report.pdf"
    source.write_text("first", encoding="utf-8")
    timestamp = 1_707_955_200  # 2024-02-15; stable across supported time zones.
    os.utime(source, (timestamp, timestamp))
    manager = FileManager(config)

    planned, errors = manager.organize()
    assert not errors
    assert planned[0].status == "planned"
    assert source.is_file()

    applied, errors = manager.organize(apply=True)
    assert not errors
    destination = Path(applied[0].destination)
    assert destination.parts[-4:-1] == ("documents", "2024", "02")
    assert destination.read_text(encoding="utf-8") == "first"
    assert not source.exists()

    second = tmp_path / "inbox" / "report.pdf"
    second.write_text("second", encoding="utf-8")
    os.utime(second, (timestamp, timestamp))
    collision_actions, errors = manager.organize(apply=True)
    assert not errors
    collision = Path(collision_actions[0].destination)
    assert collision.name == "report (1).pdf"
    assert destination.read_text(encoding="utf-8") == "first"
    assert collision.read_text(encoding="utf-8") == "second"


def test_metadata_inventory_detects_new_updated_and_missing_files(tmp_path):
    config = load_config(make_config(tmp_path))
    source = tmp_path / "important" / "notes.txt"
    source.write_text("one", encoding="utf-8")
    manager = FileManager(config)

    first, errors = manager.scan_metadata()
    assert not errors
    assert first["new"] == 1
    record = next(item for item in manager.metadata.records() if item["path"] == str(source))
    assert record["category"] == "documents"
    assert len(record["sha256"]) == 64

    source.write_text("two and changed", encoding="utf-8")
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1_000_000))
    second, errors = manager.scan_metadata()
    assert not errors
    assert second["updated"] >= 1

    source.unlink()
    third, errors = manager.scan_metadata()
    assert not errors
    assert third["missing"] == 1


def test_backup_preserves_previous_destination_version(tmp_path):
    config = load_config(make_config(tmp_path))
    source = tmp_path / "important" / "critical.txt"
    source.write_text("version one", encoding="utf-8")
    manager = FileManager(config)

    first, errors = manager.backup(apply=True)
    assert not errors
    assert [item.operation for item in first] == ["backup"]
    destination = tmp_path / "backup" / "critical.txt"
    assert destination.read_text(encoding="utf-8") == "version one"

    source.write_text("version two", encoding="utf-8")
    os.utime(source, ns=(source.stat().st_atime_ns, destination.stat().st_mtime_ns + 2_000_000))
    second, errors = manager.backup(apply=True)
    assert not errors
    assert [item.operation for item in second] == ["backup-version", "backup"]
    assert destination.read_text(encoding="utf-8") == "version two"
    versions = list((tmp_path / "backup" / ".versions").rglob("critical.txt"))
    assert len(versions) == 1
    assert versions[0].read_text(encoding="utf-8") == "version one"


def test_sync_skips_newer_destination_and_never_deletes_extra_files(tmp_path):
    config = load_config(make_config(tmp_path))
    source_root = tmp_path / "sync-source"
    destination_root = tmp_path / "sync-destination"
    source = source_root / "shared.txt"
    destination = destination_root / "shared.txt"
    extra = destination_root / "destination-only.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("newer destination", encoding="utf-8")
    extra.write_text("keep", encoding="utf-8")
    os.utime(source, ns=(source.stat().st_atime_ns, 1_000_000_000))
    os.utime(destination, ns=(destination.stat().st_atime_ns, 2_000_000_000))
    manager = FileManager(config)

    actions, errors = manager.synchronize(apply=True)
    assert not errors
    assert actions[0].status == "skipped"
    assert destination.read_text(encoding="utf-8") == "newer destination"
    assert extra.read_text(encoding="utf-8") == "keep"

    new_source = source_root / "new.txt"
    new_source.write_text("copy me", encoding="utf-8")
    actions, errors = manager.synchronize(apply=True)
    assert not errors
    assert any(item.status == "applied" for item in actions)
    assert (destination_root / "new.txt").read_text(encoding="utf-8") == "copy me"
    assert extra.exists()

    same_source = source_root / "same-metadata.txt"
    same_destination = destination_root / "same-metadata.txt"
    same_source.write_text("aaaa", encoding="utf-8")
    same_destination.write_text("bbbb", encoding="utf-8")
    identical_time = 3_000_000_000
    os.utime(same_source, ns=(identical_time, identical_time))
    os.utime(same_destination, ns=(identical_time, identical_time))
    actions, errors = manager.synchronize(apply=True)
    assert not errors
    assert any(Path(item.source) == same_source for item in actions)
    assert same_destination.read_text(encoding="utf-8") == "aaaa"
    versions = list(
        (destination_root / ".versions").rglob("same-metadata.txt")
    )
    assert len(versions) == 1
    assert versions[0].read_text(encoding="utf-8") == "bbbb"


def test_run_cycle_writes_reports_and_notifications(tmp_path):
    config = load_config(make_config(tmp_path))
    (tmp_path / "inbox" / "photo.jpg").write_bytes(b"image")
    manager = FileManager(config)

    report = manager.run_cycle(apply=False)
    assert report["status"] == "ok"
    assert report["mode"] == "dry-run"
    assert report["summary"]["planned"] == 1
    assert Path(report["reports"]["json"]).is_file()
    assert Path(report["reports"]["markdown"]).is_file()
    assert (tmp_path / "reports" / "latest.json").is_file()
    notifications = (tmp_path / "state" / "notifications.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"status": "ok"' in notifications
    assert (tmp_path / "inbox" / "photo.jpg").exists()
