"""Pinned source acquisition and independent, correction-based adjudication.

Never imports or executes acquired projects. Gold patches stay outside runtime input.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


# Reviewed against original reports, production patches and added regression tests.
# This table is adjudication, not predictive runtime logic.
ADJUDICATION = {
    "psf__requests-1963": ("development", "BEHAVIORAL_MISMATCH", "requests/sessions.py", "resolve_redirects", "Redirect iterations copy the initial request instead of the latest prepared request.", "303 followed by 307 preserves GET after the first redirect.", "Later redirect restores original POST.", ["FIX_PRODUCER", "CORRECT_CONTROL_FLOW"], 1962),
    "psf__requests-2317": ("development", "DATA_OR_RETURN_CONTRACT", "requests/sessions.py", "request", "Bytes method is stringified, not decoded to native text.", "Bytes GET is sent as GET.", "Method becomes literal b'GET' and receives 404.", ["FIX_PRODUCER", "FIX_CONSUMER_CONTRACT", "CORRECT_ARGUMENT"], 2316),
    "psf__requests-2674": ("development", "EXCEPTION_FAILURE", "requests/adapters.py", "send", "ClosedPoolError is not translated at the adapter API boundary.", "Closed pools produce requests ConnectionError.", "urllib3 ClosedPoolError escapes the public API.", ["ERROR_HANDLING", "FIX_CONSUMER_CONTRACT"], None),
    "psf__requests-3362": ("development", "DATA_OR_RETURN_CONTRACT", "requests/utils.py", "stream_decode_response_unicode", "Unknown declared encoding bypasses unicode decoding instead of falling back to apparent encoding.", "decode_unicode=True yields text even when encoding is None.", "iter_content yields bytes while text yields unicode.", ["FIX_PRODUCER", "CORRECT_RETURN_VALUE", "CORRECT_CONTROL_FLOW"], None),
    "psf__requests-2148": ("holdout", "EXCEPTION_FAILURE", "requests/models.py", "iter_content", "The streaming exception boundary does not wrap socket.error.", "Socket reset becomes requests ConnectionError.", "Raw socket error escapes while consuming response content.", ["ERROR_HANDLING", "FIX_CONSUMER_CONTRACT"], None),
    "psf__requests-863": ("holdout", "DATA_OR_RETURN_CONTRACT", "requests/models.py", "register_hook", "A collection of hooks is appended as a single non-callable element.", "Every registered hook is callable.", "List of hooks is nested inside the hook list.", ["FIX_PRODUCER", "FIX_CONSUMER_CONTRACT", "CORRECT_CONTROL_FLOW"], 785),
    "marshmallow-code__marshmallow-1359": ("development", "EXCEPTION_FAILURE", "src/marshmallow/fields.py", "_bind_to_schema", "Nested DateTime reads opts from its container rather than root schema.", "DateTime inside List or Tuple inherits schema date format.", "Container fields lack schema.opts and binding fails.", ["FIX_PRODUCER", "FIX_CONSUMER_CONTRACT", "CORRECT_ARGUMENT"], 1357),
    "marshmallow-code__marshmallow-1343": ("holdout", "EXCEPTION_FAILURE", "src/marshmallow/schema.py", "_invoke_field_validators", "Validators index invalid nested data but catch only KeyError, not TypeError.", "Invalid nested input produces ValidationError.", "None data indexing raises uncaught TypeError.", ["ERROR_HANDLING", "FIX_CONSUMER_CONTRACT", "VALIDATE_BOUNDARY"], 1342),
    "pytest-dev__pytest-11143": ("development", "EXCEPTION_FAILURE", "src/_pytest/assertion/rewrite.py", "run", "Any leading AST Constant is treated as a string docstring, including numbers.", "A test module beginning with a numeric expression can be rewritten.", "Numeric constant is passed to string-oriented rewrite-disable handling.", ["CORRECT_CONTROL_FLOW", "VALIDATE_BOUNDARY", "FIX_CONSUMER_CONTRACT"], 11140),
    "pytest-dev__pytest-5413": ("development", "BEHAVIORAL_MISMATCH", "src/_pytest/_code/code.py", "__str__", "ExceptionInfo string formatting appends source location unlike expected exception representation.", "ExceptionInfo str and repr provide consistent exception description.", "str includes a location prefix and differs from expected presentation.", ["FIX_PRODUCER", "CORRECT_RETURN_VALUE"], None),
    "pytest-dev__pytest-7373": ("development", "BEHAVIORAL_MISMATCH", "src/_pytest/mark/evaluate.py", "cached_eval", "Expression results are cached by text without the evaluation globals.", "Identical skipif expressions in different modules evaluate against each module's globals.", "Cached result from the first module is reused in the second.", ["FIX_PRODUCER", "CORRECT_CONTROL_FLOW", "CORRECT_RETURN_VALUE"], 7360),
    "pytest-dev__pytest-8365": ("holdout", "CONFIGURATION_BEHAVIOR", "src/_pytest/tmpdir.py", "getbasetemp", "User-derived temp directory names are not guarded for invalid platform characters.", "Temporary directory creation falls back to a usable name.", "Invalid username characters cause directory creation to fail.", ["ERROR_HANDLING", "CORRECT_CONFIGURATION", "VALIDATE_BOUNDARY"], None),
}

OWNERS = {
    "psf__requests-1963": "SessionRedirectMixin",
    "psf__requests-2317": "Session",
    "psf__requests-2674": "HTTPAdapter",
    "psf__requests-2148": "Response",
    "psf__requests-863": "Request",
    "marshmallow-code__marshmallow-1359": "DateTime",
    "marshmallow-code__marshmallow-1343": "BaseSchema",
    "pytest-dev__pytest-11143": "AssertionRewriter",
    "pytest-dev__pytest-5413": "ExceptionInfo",
    "pytest-dev__pytest-8365": "TempPathFactory",
}


def symbol_locations(tree: ast.AST, symbol: str, owner: str | None) -> list[dict]:
    nodes = [tree] if owner is None else [node for node in ast.walk(tree)
                                        if isinstance(node, ast.ClassDef) and node.name == owner]
    return [{"line": node.lineno, "end_line": node.end_lineno}
            for parent in nodes for node in ast.walk(parent)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_hash(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        value.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return value.hexdigest()


def projection_path(root: Path, relative: PurePosixPath) -> Path:
    if (relative.is_absolute() or ".." in relative.parts
            or any("\\" in part or ":" in part for part in relative.parts)):
        raise ValueError("Caminho de arquivo de origem inseguro.")
    dest = root.joinpath(*relative.parts)
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError("Arquivo de origem escapou da raiz.")
    return dest


def main() -> None:
    cache = Path(".orchestrator/external-v12")
    dataset = json.loads((cache / "dataset.json").read_text(encoding="utf-8"))
    rows = {row["instance_id"]: row for row in dataset["rows"]}
    target = Path("tests/data/predictive/external_v12")
    previous = None
    if (target / "manifest.json").exists():
        previous = json.loads((target / "manifest.json").read_text())
    target.mkdir(parents=True, exist_ok=True, mode=0o777)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cases = []
    for case_id, annotation in ADJUDICATION.items():
        split, category, path, symbol, cause, expected, observed, strategies, issue = annotation
        row = rows[case_id]
        root = cache / "snapshots" / case_id
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        archive = cache / f"{case_id}.tar.gz"
        if not archive.exists():
            archive.write_bytes(opener.open(url, timeout=120).read())
        if not root.exists():
            root.mkdir(parents=True, mode=0o777)
            total = 0
            with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
                for member in tar:
                    relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                    if not member.isfile() or relative.suffix != ".py":
                        continue
                    if relative.is_absolute() or ".." in relative.parts or member.size > 2_000_000:
                        raise ValueError("Arquivo de origem fora dos limites.")
                    total += member.size
                    if total > 64_000_000:
                        raise ValueError("Orcamento de snapshot excedido.")
                    dest = projection_path(root, relative)
                    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
                    dest.write_bytes(tar.extractfile(member).read())
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        owner = OWNERS.get(case_id)
        locations = symbol_locations(tree, symbol, owner)
        if not locations:
            raise ValueError(f"Origem adjudicada ausente: {case_id}")
        repo_url = f"https://github.com/{row['repo']}"
        case = {
            "schema_version": 12, "id": case_id, "split": split, "category": category,
            "repo": row["repo"], "base_commit": row["base_commit"],
            "source_archive_url": url, "source_archive_sha256": digest(archive.read_bytes()),
            "source_python_sha256": source_hash(root),
            "dataset_revision": dataset["revision"],
            "problem_formulations": [row["problem_statement"]],
            "issue_url": f"{repo_url}/issues/{issue}" if issue else f"{repo_url}/pull/{case_id.rsplit('-', 1)[1]}",
            "corrective_pr_url": f"{repo_url}/pull/{case_id.rsplit('-', 1)[1]}",
            "gold_patch_sha256": digest(row["patch"].encode()),
            "gold_test_patch_sha256": digest(row["test_patch"].encode()),
            "oracle": {"root_path": path, "root_symbol": f"{owner}.{symbol}" if owner else symbol,
                       "source_locations": locations, "root_cause": cause,
                       "expected_behavior": expected, "observed_behavior": observed,
                       "repair_strategies": strategies, "target_paths": [path],
                       "abstention_expected": False, "import_causal": False,
                       "adjudication_basis": "Original issue, pre-fix AST, production correction and added regression test; no model output used.",
                       "limitation": "Location/strategy proxies, not full semantic entailment; no upstream tests executed."},
        }
        cases.append(case)
        print(case_id, len(locations), case["source_python_sha256"], flush=True)
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True, indent=2).encode()
    if previous and previous.get("adjudication_revision") == 2:
        if digest(payload + b"\n") != previous["cases_sha256"]:
            raise ValueError("Reproducao divergiu do corpus selado; nao sobrescrever.")
        print("Snapshots reproduzidos; seal preservado.")
        return
    (target / "cases.json").write_bytes(payload + b"\n")
    (target / "manifest.json").write_text(json.dumps({
        "version": 12, "adjudication_revision": 2,
        "previous_pre_tuning_seal": previous.get("cases_sha256") if previous else None,
        "revision_reason": "Disambiguate methods sharing a name using correction's owning class, before any engine changes or live evaluation.",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "engine_commit": "601b4e697a0d0ee6d79ca716939e37a41d293902",
        "dataset": "princeton-nlp/SWE-bench_Lite", "dataset_revision": dataset["revision"],
        "cases_sha256": digest(payload + b"\n"), "independent_defects": len(cases),
        "formulations": sum(len(case["problem_formulations"]) for case in cases),
        "development": sum(case["split"] == "development" for case in cases),
        "holdout": sum(case["split"] == "holdout" for case in cases),
        "repositories": sorted({case["repo"] for case in cases}),
        "snapshot_policy": "Python source-only archive projection, pinned pre-fix commit; no imports, installs or project execution.",
        "historical_requests_5667": "excluded from independent benchmark populations",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
