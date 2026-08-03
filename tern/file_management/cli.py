from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigurationError, load_config
from .manager import FileManager


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _action_result(name: str, actions: list, errors: List[str], apply: bool) -> Dict[str, Any]:
    return {
        "ok": not errors,
        "operation": name,
        "mode": "apply" if apply else "dry-run",
        "actions": [item.as_dict() for item in actions],
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe, policy-driven file organization, inventory, backup, and sync."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON policy file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate configuration and source directories")
    subparsers.add_parser("scan", help="refresh the metadata inventory")
    for name in ("organize", "backup", "sync", "run"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--apply",
            action="store_true",
            help="apply changes; without this flag the command is a dry-run",
        )
    watch = subparsers.add_parser("watch", help="run organization cycles continuously")
    watch.add_argument(
        "--apply",
        action="store_true",
        help="apply changes in every cycle; default is continuous dry-run",
    )
    watch.add_argument(
        "--interval",
        type=int,
        help="override interval_seconds (minimum 10)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config.resolve(strict=False))
        manager = FileManager(config)
        if args.command == "validate":
            result = manager.validate_runtime()
            _print(result)
            return 0 if result["ok"] else 1
        if args.command == "scan":
            metadata, errors = manager.scan_metadata()
            _print({"ok": not errors, "metadata": metadata, "errors": errors})
            return 0 if not errors else 1
        if args.command == "organize":
            actions, errors = manager.organize(apply=args.apply)
            result = _action_result("organize", actions, errors, args.apply)
        elif args.command == "backup":
            actions, errors = manager.backup(apply=args.apply)
            result = _action_result("backup", actions, errors, args.apply)
        elif args.command == "sync":
            actions, errors = manager.synchronize(apply=args.apply)
            result = _action_result("sync", actions, errors, args.apply)
        elif args.command == "run":
            result = manager.run_cycle(apply=args.apply)
            _print(result)
            return 0 if result["status"] == "ok" else 1
        elif args.command == "watch":
            interval = args.interval or config.interval_seconds
            if interval < 10:
                raise ConfigurationError("watch interval must be at least 10 seconds")
            try:
                while True:
                    result = manager.run_cycle(apply=args.apply)
                    if result["status"] != "ok":
                        print("cycle completed with errors", file=sys.stderr, flush=True)
                    time.sleep(interval)
            except KeyboardInterrupt:
                return 0
        else:
            raise ConfigurationError(f"unsupported command: {args.command}")
        _print(result)
        return 0 if result["ok"] else 1
    except (ConfigurationError, OSError) as exc:
        _print({"ok": False, "error": type(exc).__name__, "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
