from __future__ import annotations

from pathlib import Path, PurePosixPath

from ...security import PathPolicy

_PROTECTED = {".git", ".ssh", ".aws", ".azure", "credentials", "secrets", "node_modules", ".venv"}
_SUFFIXES = {".py", ".json", ".toml", ".cfg", ".ini", ".txt"}


def scoped_file(root: Path, raw: str) -> Path:
    parts = PurePosixPath(raw).parts
    if (not raw or "\\" in raw or ":" in raw or "\0" in raw
            or raw.startswith("/") or any(part in {"..", ".", ""} for part in raw.split("/"))):
        raise ValueError("unsafe relative path")
    if any(part.casefold() in _PROTECTED or part.casefold().startswith(".env") for part in parts):
        raise ValueError("protected path")
    if any(word in parts[-1].casefold() for word in ("credential", "secret", "token", "id_rsa", "id_ed25519")):
        raise ValueError("sensitive path")
    path = PathPolicy((root,)).resolve(str(root / raw))
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("symlink or junction is forbidden")
    if not path.is_file() or path.suffix.casefold() not in _SUFFIXES:
        raise ValueError("unsupported input file")
    return path


def copy_scoped_snapshot(root: Path, destination: Path, files: tuple[str, ...], *, max_bytes: int = 2_097_152) -> None:
    root = root.resolve(strict=True)
    destination = destination.resolve(strict=True)
    if destination == root or destination.is_relative_to(root):
        raise ValueError("snapshot must be outside original tree")
    if any(destination.iterdir()):
        raise ValueError("snapshot destination must be empty")
    if len(files) > 32 or len(set(files)) != len(files):
        raise ValueError("snapshot file budget exceeded")
    contents = []
    total = 0
    for relative in files:
        content = scoped_file(root, relative).read_bytes()
        total += len(content)
        if total > max_bytes or b"\0" in content:
            raise ValueError("snapshot byte budget or binary restriction")
        content.decode("utf-8")
        contents.append((relative, content))
    for relative, content in contents:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
