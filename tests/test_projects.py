from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.config import load_settings
from tern.orchestrator.projects import (
    IGNORED_DIRECTORIES,
    ProjectRegistry,
    normalize_technical_transcript,
)
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tool_progress import ToolProgressTracker
from tern.orchestrator.tools import ToolRegistry
from tern.orchestrator.voice.logging import VoiceLogger


class Jobs:
    def list(self):
        return []


class Codex:
    timeout = 30

    def __init__(self, project: Path | None = None):
        self.project = project
        self.jobs = Jobs()

    def shared_project(self):
        return self.project

    def claim_completed_results(self):
        return []


def project(base: Path, name: str, *, python: bool = True) -> Path:
    root = base / name
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "README.md").write_text(name, encoding="utf-8")
    if python:
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / "tests").mkdir()
    return root


def registry(base: Path, roots: tuple[Path, ...], codex=None) -> ProjectRegistry:
    return ProjectRegistry(PathPolicy(roots), base / "state", codex=codex)


def test_discovers_configured_and_nested_git_projects(tmp_path):
    root = project(tmp_path, "tern")
    nested = project(root, "plugin")
    projects = registry(tmp_path, (root,)).projects()
    assert {item["root"] for item in projects} == {str(root.resolve()), str(nested.resolve())}


def test_readme_alone_is_not_a_nested_project(tmp_path):
    root = project(tmp_path, "tern")
    folder = root / "notes"
    folder.mkdir()
    (folder / "README.md").write_text("notes", encoding="utf-8")
    assert str(folder.resolve()) not in {item["root"] for item in registry(tmp_path, (root,)).projects()}


def test_markers_and_type_are_persisted(tmp_path):
    root = project(tmp_path, "tern")
    item = registry(tmp_path, (root,)).projects()[0]
    assert item["type"] == "python"
    assert {".git", "pyproject.toml", "tests"}.issubset(item["markers"])


def test_default_aliases_are_normalized_and_unique(tmp_path):
    root = project(tmp_path, "tern")
    item = registry(tmp_path, (root,)).projects()[0]
    assert item["aliases"] == sorted(set(item["aliases"]))
    assert {"jarvis", "assistente", "orquestrador", "tern"}.issubset(item["aliases"])


def test_aliases_can_be_configured_in_persistent_registry(tmp_path):
    root = project(tmp_path, "tern")
    values = registry(tmp_path, (root,))
    state = values.read()
    state["projects"][0]["aliases"].extend(["Meu Assistente", "meu assistente"])
    values.path.write_text(json.dumps(state), encoding="utf-8")
    refreshed = values.refresh()
    aliases = next(item for item in refreshed["projects"] if item["id"] == "tern")["aliases"]
    assert aliases.count("meu assistente") == 1
    assert values.resolve(query="corrija meu assistente")["project_id"] == "tern"


def test_active_project_survives_restart(tmp_path):
    tern = project(tmp_path, "tern")
    llama = project(tmp_path, "llama.cpp", python=False)
    first = registry(tmp_path, (tern, llama))
    assert first.use("llama")["project_id"] == "llama.cpp"
    second = registry(tmp_path, (tern, llama))
    assert second.active()["project"]["id"] == "llama.cpp"


def test_resolve_explicit_path_and_alias(tmp_path):
    tern = project(tmp_path, "tern")
    values = registry(tmp_path, (tern,))
    assert values.resolve(path_hint=str(tern / "tests"))["matched_by"] == "path"
    alias = values.resolve(query="Corrija o Jarvis")
    assert alias["root"] == str(tern.resolve()) and alias["matched_by"] == "alias"


def test_codex_thread_project_precedes_active_project(tmp_path):
    tern = project(tmp_path, "tern")
    llama = project(tmp_path, "llama.cpp", python=False)
    values = registry(tmp_path, (tern, llama), Codex(tern))
    values.use("llama")
    assert values.resolve(query="continue a tarefa")["root"] == str(tern.resolve())
    assert values.resolve(query="continue a tarefa")["matched_by"] == "codex_thread"


def test_explicit_project_name_precedes_codex_thread(tmp_path):
    tern = project(tmp_path, "tern")
    llama = project(tmp_path, "llama.cpp", python=False)
    values = registry(tmp_path, (tern, llama), Codex(tern))
    result = values.resolve(query="Veja o llama.cpp")
    assert result["project_id"] == "llama.cpp" and result["matched_by"] == "alias"


def test_multiple_named_projects_are_reported_as_ambiguous(tmp_path):
    tern = project(tmp_path, "tern")
    llama = project(tmp_path, "llama.cpp", python=False)
    result = registry(tmp_path, (tern, llama)).resolve(query="compare tern e llama")
    assert result["error"] == "ambiguous_project"
    assert len(result["alternatives"]) == 2


def test_external_path_is_refused(tmp_path):
    root = project(tmp_path, "tern")
    outside = tmp_path / "outside"
    outside.mkdir()
    result = registry(tmp_path, (root,)).resolve(path_hint=str(outside))
    assert result["error"] == "project_path_not_allowed"


def test_exact_partial_and_description_file_search(tmp_path):
    root = project(tmp_path, "tern")
    voice = root / "tern" / "orchestrator" / "voice"
    voice.mkdir(parents=True)
    (voice / "windows_speech.py").write_text("class DanielProvider: pass", encoding="utf-8")
    (root / ".env.example").write_text("VOICE_WINDOWS_RATE=1.5", encoding="utf-8")
    values = registry(tmp_path, (root,))
    exact = values.find_files(project_id="tern", query="windows_speech.py")
    partial = values.find_files(project_id="tern", query="windows speech")
    described = values.find_files(project_id="tern", query="configuracao do Daniel")
    assert exact["results"][0]["path"].endswith("windows_speech.py")
    assert partial["results"][0]["path"].endswith("windows_speech.py")
    assert {item["path"] for item in described["results"]}.intersection(
        {".env.example", "tern/orchestrator/voice/windows_speech.py"}
    )


def test_file_types_filter_results(tmp_path):
    root = project(tmp_path, "tern")
    (root / "config.py").write_text("x=1", encoding="utf-8")
    (root / "config.md").write_text("x", encoding="utf-8")
    result = registry(tmp_path, (root,)).find_files(
        project_id="tern", query="config", file_types=["py"]
    )
    assert result["results"] and all(item["path"].endswith(".py") for item in result["results"])


def test_incremental_index_updates_renamed_file(tmp_path):
    root = project(tmp_path, "tern")
    old = root / "old_config.py"
    old.write_text("x=1", encoding="utf-8")
    values = registry(tmp_path, (root,))
    values.refresh_index("tern")
    old.rename(root / "new_config.py")
    result = values.find_files(project_id="tern", query="new_config.py")
    assert result["results"][0]["path"] == "new_config.py"
    index = json.loads((tmp_path / "state" / "project-indexes" / "tern.json").read_text(encoding="utf-8"))
    assert "old_config.py" not in {item["path"] for item in index["files"]}


def test_corrupt_index_is_rebuilt(tmp_path):
    root = project(tmp_path, "tern")
    (root / "config.py").write_text("x=1", encoding="utf-8")
    values = registry(tmp_path, (root,))
    values.refresh_index("tern")
    index = tmp_path / "state" / "project-indexes" / "tern.json"
    index.write_text("{broken", encoding="utf-8")
    result = values.find_files(project_id="tern", query="config.py")
    assert result["results"][0]["path"] == "config.py"
    assert json.loads(index.read_text(encoding="utf-8"))["project_id"] == "tern"


def test_index_ignores_caches_models_audio_and_sensitive_files(tmp_path):
    root = project(tmp_path, "tern")
    for directory in (".venv", "node_modules", "__pycache__", "build", "dist", "target", "models", ".orchestrator", ".cache"):
        folder = root / directory
        folder.mkdir(exist_ok=True)
        (folder / "secret.py").write_text("secret", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (root / "voice.wav").write_bytes(b"audio")
    (root / "checkpoint.safetensors").write_bytes(b"weights")
    values = registry(tmp_path, (root,))
    values.refresh_index("tern")
    index = json.loads((tmp_path / "state" / "project-indexes" / "tern.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in index["files"]}
    assert not {".env", "voice.wav", "checkpoint.safetensors"}.intersection(paths)
    assert not any(Path(path).parts[0] in IGNORED_DIRECTORIES for path in paths)
    assert "secret" not in json.dumps(index)


def test_symlink_escaping_root_is_not_indexed(tmp_path):
    root = project(tmp_path, "tern")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "linked"
    junction = False
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("symlink indisponivel")
        created = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
        )
        if created.returncode:
            pytest.skip("symlink e junction indisponiveis")
        junction = True
    try:
        values = registry(tmp_path, (root,))
        values.refresh_index("tern")
        index = json.loads((tmp_path / "state" / "project-indexes" / "tern.json").read_text(encoding="utf-8"))
        assert not any(item["path"].startswith("linked/") for item in index["files"])
    finally:
        if junction and link.exists():
            os.rmdir(link)


def test_removed_project_is_dropped_on_refresh(tmp_path):
    tern = project(tmp_path, "tern")
    extra = project(tmp_path, "extra")
    values = registry(tmp_path, (tern, extra))
    shutil.rmtree(extra)
    assert str(extra.resolve()) not in {item["root"] for item in values.projects()}


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("revise o código ex", "revise o Codex"),
        ("abra o projeto terne", "abra o projeto Tern"),
        ("veja o lama ponto cpp", "veja o llama.cpp"),
        ("corrija o jarves", "corrija o Jarvis"),
    ],
)
def test_contextual_stt_corrections(original, expected):
    assert normalize_technical_transcript(original) == expected


def test_stt_does_not_replace_ordinary_sentence():
    original = "a lama cobriu a estrada"
    assert normalize_technical_transcript(original) == original


def test_original_and_routing_transcripts_can_be_logged(tmp_path):
    logger = VoiceLogger(tmp_path / "voice.jsonl", debug_transcripts=True)
    logger.write(
        "transcription_routing",
        transcript_original="código ex",
        transcript_normalized="Codex",
    )
    record = json.loads(logger.path.read_text(encoding="utf-8"))
    assert record["transcript_original"] == "código ex"
    assert record["transcript_normalized"] == "Codex"


def test_original_routing_transcript_is_preserved_without_debug(tmp_path):
    logger = VoiceLogger(tmp_path / "voice.jsonl", debug_transcripts=False)
    logger.write(
        "transcription_routing",
        transcript_original="código ex",
        transcript_normalized="Codex",
    )
    record = json.loads(logger.path.read_text(encoding="utf-8"))
    assert record["transcript_original"] == "código ex"
    assert "transcript_normalized" not in record


def test_tool_progress_blocks_third_equivalent_call():
    tracker = ToolProgressTracker()
    arguments = {"path": "D:/tern"}
    result = {"ok": True, "entries": [{"path": "config.py", "size": 10}]}
    tracker.record("filesystem_list", arguments, result)
    tracker.record("filesystem_list", arguments, result)
    assert tracker.should_block("filesystem_list", arguments)


def test_tool_progress_allows_distinct_calls_and_new_paths():
    tracker = ToolProgressTracker()
    tracker.record("filesystem_list", {"path": "D:/tern"}, {"ok": True, "path": "a"})
    assert not tracker.should_block("filesystem_list", {"path": "D:/tern/tests"})
    second = tracker.record("filesystem_list", {"path": "D:/tern/tests"}, {"ok": True, "path": "b"})
    assert second.progress


class SequenceClient:
    def __init__(self, values):
        self.values = iter(values)

    def chat(self, _messages, **_kwargs):
        return next(self.values)


def _tool_response(identifier: str, path: Path):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": identifier,
                            "type": "function",
                            "function": {
                                "name": "filesystem_list",
                                "arguments": json.dumps({"path": str(path)}),
                            },
                        }
                    ],
                }
            }
        ]
    }


def test_supervisor_prevents_third_equivalent_execution(tmp_path):
    root = project(tmp_path, "tern")
    state = tmp_path / "state"
    settings = load_settings({"MODEL_ALLOWED_ROOTS": str(root), "MODEL_STATE_DIR": str(state)})
    tools = ToolRegistry(
        policy=PathPolicy((root,)),
        logger=ActionLogger(state / "actions.jsonl"),
        codex=Codex(root),
        max_output_bytes=131072,
    )
    client = SequenceClient(
        [
            _tool_response("one", root),
            _tool_response("two", root),
            _tool_response("three", root),
            {"choices": [{"message": {"role": "assistant", "content": "Reformulei."}}]},
        ]
    )
    result = Supervisor(settings, client, tools).run("Liste sem repetir")
    assert result["ok"] and result["tool_calls"] == 3
    records = [json.loads(line) for line in tools.logger.path.read_text(encoding="utf-8").splitlines()]
    executed = [item for item in records if item.get("event") == "tool_result" and item.get("tool") == "filesystem_list"]
    assert len(executed) == 2
    assert any(item.get("event") == "tool_loop_prevented" for item in records)


def test_delegate_normalization_uses_alias_not_home_directory(tmp_path):
    root = project(tmp_path, "tern")
    codex = Codex(root)
    tools = ToolRegistry(
        policy=PathPolicy((root,)),
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=codex,
        max_output_bytes=131072,
    )
    normalized = tools._normalize_arguments(
        "delegate_to_codex",
        {"task": "Corrija o roteamento", "project_path": str(Path.home())},
        {"user_text": "Corrija o Jarvis", "turn_id": "one"},
    )
    assert normalized["project_path"] == str(root.resolve())
    assert normalized["task"].splitlines()[0] == f"Trabalhe no projeto {root.resolve()}."


def test_project_tools_are_small_and_read_only(tmp_path):
    root = project(tmp_path, "tern")
    tools = ToolRegistry(
        policy=PathPolicy((root,)),
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=Codex(root),
        max_output_bytes=131072,
    )
    specs = {item["function"]["name"]: item["function"] for item in tools.specs()}
    assert set(specs["resolve_project"]["parameters"]["properties"]) == {"query", "path_hint", "require_unique"}
    assert set(specs["find_project_files"]["parameters"]["properties"]) == {"project_id", "query", "file_types", "max_results"}


class FinalClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **_kwargs):
        self.messages = messages
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}


def test_qwen_project_context_is_short(tmp_path):
    root = project(tmp_path, "tern")
    settings = load_settings({"MODEL_ALLOWED_ROOTS": str(root), "MODEL_STATE_DIR": str(tmp_path / "state")})
    tools = ToolRegistry(
        policy=PathPolicy((root,)),
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=Codex(root),
        max_output_bytes=131072,
    )
    client = FinalClient()
    assert Supervisor(settings, client, tools).run("Onde esta config.py?")["ok"]
    system = client.messages[0]["content"]
    block = system.split("Project context:\n", 1)[1].split("\n\n", 1)[0]
    assert len(block) < 1000
    assert "Active project: Tern" in block
    assert "Allowed roots: 1" in block
