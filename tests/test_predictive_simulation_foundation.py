from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tern.orchestrator import cli
from tern.orchestrator.predictive.artifacts import compact_report
from tern.orchestrator.predictive.repair import RepairTarget, RepairTargetKind
from tern.orchestrator.predictive.simulation.models import (
    ChangeBudget, ObservedCandidateOutcome, OutcomeStatus, PatchProposal,
)
from tern.orchestrator.predictive.simulation.comparison import compare_observed
from tern.orchestrator.predictive.simulation.patching import QwenPatchProposalProvider, validate_patch
from tern.orchestrator.predictive.simulation.service import assess_proposal
from tern.orchestrator.predictive.simulation.sandbox import (
    DockerSandboxProvider, SandboxCapabilities, SandboxRunSpec, _sanitized_host_environment,
    detect_sandbox_provider,
)
from tern.orchestrator.predictive.simulation.workspace import copy_scoped_snapshot

IMAGE = "trusted/python@sha256:" + "a" * 64
SOURCE = "def total(x):\n    return x + 1\n\ndef unrelated():\n    return 9\n"
DIFF = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def total(x):\n-    return x + 1\n+    return x + 2\n"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text(SOURCE, encoding="utf-8")
    return root


def proposal(diff=DIFF):
    return PatchProposal("C1", "report-hash", "6b288b8", "RC1", "CORRECT_RETURN_VALUE",
        (RepairTarget("calc.py", RepairTargetKind.RETURN_SITE, symbol="total", line=2),),
        ("E1",), ("calc.py",), diff, 1, 0.8)


def test_valid_preview_is_static_and_original_bytes_unchanged(project, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("no host execution"))
    before = (project / "calc.py").read_bytes()
    outcome = assess_proposal(proposal(), project)
    assert outcome.validation.passed
    assert outcome.status is OutcomeStatus.NOT_TESTED
    assert not outcome.baseline_tests and not outcome.patched_tests
    assert not outcome.execution_authorized and outcome.requires_approval
    assert (project / "calc.py").read_bytes() == before
    assert outcome.validation.preimage_sha256 != outcome.validation.preview_sha256
    assert outcome.prediction_score == 0.8 and outcome.prediction_rank == 1


@pytest.mark.parametrize("path", ["../calc.py", "/calc.py", "C:/calc.py", "a\\calc.py", ".git/config", ".env", "secrets.json", "credentials.txt"])
def test_diff_protected_paths_rejected(project, path):
    item = replace(proposal(DIFF.replace("calc.py", path)), allowed_files=(path,))
    assert not validate_patch(item, project).passed


def test_symlink_escape_rejected(project, tmp_path):
    other = tmp_path / "outside.py"
    other.write_text(SOURCE)
    try:
        (project / "link.py").symlink_to(other)
    except OSError:
        pytest.skip("host symlink privilege unavailable")
    item = replace(proposal(DIFF.replace("calc.py", "link.py")), allowed_files=("link.py",))
    assert not validate_patch(item, project).passed
    assert other.read_text() == SOURCE


@pytest.mark.parametrize("diff", [
    DIFF.replace("x + 1", "x + 5"),
    DIFF.replace("x + 2", "("),
    DIFF.replace("@@ -1,2 +1,2 @@", "@@ -1,7 +1,2 @@"),
    DIFF.replace("+++ b/calc.py", "+++ /dev/null"),
    "GIT binary patch\n",
    DIFF + "not a diff\n",
])
def test_malformed_preimage_syntax_binary_and_deletion_rejected(project, diff):
    assert not validate_patch(proposal(diff), project).passed


def test_other_symbol_cannot_inherit_scope(project):
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -4,2 +4,2 @@\n def unrelated():\n-    return 9\n+    return 10\n"
    assert not validate_patch(proposal(diff), project).passed


def test_insertion_cannot_smuggle_new_global_code_through_return_scope(project):
    diff = DIFF.replace("+    return x + 2\n", "+    return x + 2\n+\n+raise RuntimeError('global side effect')\n").replace("+1,2 @@", "+1,4 @@")
    assert not validate_patch(proposal(diff), project).passed


def test_parameter_scope_rejects_sibling_default_edit(project):
    (project / "calc.py").write_text("def total(x=1, y=2):\n    return x + y\n")
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n-def total(x=1, y=2):\n+def total(x=1, y=3):\n     return x + y\n"
    item = replace(proposal(diff), targets=(RepairTarget(
        "calc.py", RepairTargetKind.PARAMETER, symbol="total", parameter="x", line=1),))
    assert not validate_patch(item, project).passed
    correct = replace(item, diff=diff.replace("x=1, y=3", "x=3, y=2"))
    assert validate_patch(correct, project).passed


@pytest.mark.parametrize("keyword", [False, True])
def test_callsite_scope_rejects_sibling_argument_edit(project, keyword):
    call = "foo(price=1, tax=2)" if keyword else "foo(1, 2)"
    (project / "calc.py").write_text(f"def caller():\n    return {call}\n")
    bad_call = "foo(price=1, tax=3)" if keyword else "foo(1, 3)"
    diff = f"--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def caller():\n-    return {call}\n+    return {bad_call}\n"
    item = replace(proposal(diff), strategy_kind="CORRECT_ARGUMENT", binding_slot=0,
        binding_keyword="price" if keyword else None,
        targets=(RepairTarget("calc.py", RepairTargetKind.CALL_SITE, symbol="caller", line=2),))
    assert not validate_patch(item, project).passed
    good_call = "foo(price=3, tax=2)" if keyword else "foo(3, 2)"
    assert validate_patch(replace(item, diff=diff.replace(bad_call, good_call)), project).passed


def test_method_scope_does_not_collapse_same_named_methods(project):
    (project / "calc.py").write_text("class A:\n    def run(self):\n        return 1\n\nclass B:\n    def run(self):\n        return 2\n")
    diff = "--- a/calc.py\n+++ b/calc.py\n@@ -6,2 +6,2 @@\n     def run(self):\n-        return 2\n+        return 3\n"
    item = replace(proposal(diff), targets=(RepairTarget("calc.py", RepairTargetKind.METHOD, symbol="A.run"),))
    assert not validate_patch(item, project).passed


def test_diff_budget_enforced(project):
    item = replace(proposal(), budget=ChangeBudget(max_diff_bytes=10))
    assert validate_patch(item, project).failure_codes == ("PATCH_BUDGET_EXCEEDED",)


def test_proposal_roundtrip_and_outcome_cannot_claim_unperformed_tests():
    item = proposal()
    assert PatchProposal.from_dict(item.as_dict()) == item
    with pytest.raises(ValueError, match="baseline"):
        ObservedCandidateOutcome(item.candidate_id, item.decision_report_id, item.analyzed_commit,
            1, 0.8, item.patch_sha256, validate_empty(), OutcomeStatus.OBSERVED)


def test_observed_improvement_dominates_prediction_and_requires_baseline():
    from tern.orchestrator.predictive.simulation.models import PatchValidation
    first = ObservedCandidateOutcome("C1", "report", "commit", 1, 0.9, "hash1",
        PatchValidation(True, syntax_passed=True), OutcomeStatus.OBSERVED,
        baseline_tests=(("target", "FAIL"),), patched_tests=(("target", "FAIL"),))
    second = replace(first, candidate_id="C2", prediction_rank=2, prediction_score=0.6,
                     patched_tests=(("target", "PASS"),))
    result = compare_observed((first, second))
    assert result.prediction_order == ("C1", "C2")
    assert result.observed_order == ("C2", "C1")
    assert result.recommended_candidate_id == "C2"
    assert first.prediction_score == 0.9 and second.prediction_score == 0.6
    assert not result.execution_authorized and result.requires_approval
    already_passed = replace(second, baseline_tests=(("target", "PASS"),))
    assert compare_observed((first, already_passed)).recommended_candidate_id is None


def test_infrastructure_failure_and_equivalent_evidence_are_inconclusive():
    from tern.orchestrator.predictive.simulation.models import PatchValidation
    unavailable = ObservedCandidateOutcome("C1", "report", "commit", 1, 0.9, "hash",
        PatchValidation(True), OutcomeStatus.SANDBOX_UNAVAILABLE, infrastructure_failure=True)
    assert compare_observed((unavailable,)).recommended_candidate_id is None
    one = replace(unavailable, status=OutcomeStatus.OBSERVED, infrastructure_failure=False,
        validation=PatchValidation(True, syntax_passed=True),
        baseline_tests=(("target", "FAIL"),), patched_tests=(("target", "PASS"),))
    two = replace(one, candidate_id="C2", prediction_rank=2)
    assert compare_observed((one, two)).recommended_candidate_id is None


def validate_empty():
    from tern.orchestrator.predictive.simulation.models import PatchValidation
    return PatchValidation(True)


def test_snapshot_is_independent_and_excludes_unlisted_secret(project, tmp_path):
    (project / ".env").write_text("fake secret")
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    copy_scoped_snapshot(project, one, ("calc.py",))
    copy_scoped_snapshot(project, two, ("calc.py",))
    (one / "calc.py").write_text("changed")
    assert (two / "calc.py").read_text() == SOURCE
    assert (project / "calc.py").read_text() == SOURCE
    assert not (one / ".env").exists()
    with pytest.raises((ValueError, PermissionError)):
        copy_scoped_snapshot(project, one, (".env",))


def test_snapshot_budget_and_location(project, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="budget"):
        copy_scoped_snapshot(project, dest, ("calc.py",), max_bytes=1)
    with pytest.raises(ValueError, match="outside"):
        copy_scoped_snapshot(project, project, ("calc.py",))


def test_provider_requires_digest_and_never_pulls(tmp_path):
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker)
    assert not provider.self_test().capabilities.ready
    with pytest.raises(ValueError, match="digest"):
        SandboxRunSpec(tmp_path, ("python", "-V"), image="python:latest")
    provider = DockerSandboxProvider(docker, image=IMAGE)
    command = provider.build_command(SandboxRunSpec(tmp_path, ("python", "-V")), "controlled")
    assert "--pull" in command and "never" in command
    assert "--entrypoint" in command and "/bin/sh" in command
    assert "--memory-swap" in command and "--ulimit" in command
    with pytest.raises(ValueError, match="override"):
        provider.build_command(SandboxRunSpec(tmp_path, ("python", "-V"), image="trusted/python@sha256:" + "b" * 64), "controlled")


@pytest.mark.parametrize("key", ["PATH", "HOME", "PYTHONPATH", "LD_PRELOAD", "PYTEST_ADDOPTS", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID"])
def test_environment_is_allowlist(tmp_path, key):
    with pytest.raises(ValueError, match="allowlisted"):
        SandboxRunSpec(tmp_path, ("python", "-V"), environment=((key, "fake"),))


def test_host_environment_does_not_forward_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("DOCKER_HOST", "tcp://unsafe:2375")
    env = _sanitized_host_environment(tmp_path / "docker.exe")
    assert "GITHUB_TOKEN" not in env and "DOCKER_HOST" not in env


def test_flags_alone_do_not_promote_and_cleanup_daemon_error_is_not_absence(tmp_path, monkeypatch):
    caps = SandboxCapabilities("docker", True, True, True, True, True, True, True)
    assert not caps.ready
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker, image=IMAGE)
    monkeypatch.setattr(provider, "_control", lambda *_a: subprocess.CompletedProcess([], 1, b"", b"daemon offline"))
    assert not provider._container_absent("controlled")


def test_output_stream_is_bounded():
    buffer, truncated = bytearray(), [False]
    DockerSandboxProvider._read_stream(io.BytesIO(b"x" * 50000), 1024, buffer, truncated)
    assert len(buffer) == 1024 and truncated[0]


def test_timeout_removes_container_and_does_not_promote(tmp_path, monkeypatch):
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker, image=IMAGE)
    controls = []
    class Process:
        stdout = io.BytesIO(b"bounded")
        stderr = io.BytesIO()
        killed = False
        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("docker", timeout)
            return -1
        def poll(self):
            return -1 if self.killed else None
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(provider, "_control", lambda *args: (
        controls.append(args) or subprocess.CompletedProcess([], 0, b"", b"")
    ))
    result = provider._run_unchecked(SandboxRunSpec(tmp_path, ("python", "-V")))
    assert result.timed_out and result.cleanup_verified and process.killed
    assert controls[0][:2] == ("rm", "--force")
    assert not provider.capabilities().ready


def test_launch_exception_still_attempts_cleanup_and_reports_infrastructure(tmp_path, monkeypatch):
    docker = tmp_path / "docker.exe"
    docker.touch()
    provider = DockerSandboxProvider(docker, image=IMAGE)
    controls = []
    def fail(*_args, **_kwargs):
        raise OSError("controlled failure")
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(provider, "_control", lambda *args: (
        controls.append(args) or subprocess.CompletedProcess([], 1, b"", b"daemon offline")
    ))
    result = provider._run_unchecked(SandboxRunSpec(tmp_path, ("python", "-V")))
    assert not result.executed and result.infrastructure_failure
    assert not result.cleanup_verified and not provider.capabilities().ready
    assert controls[0][:2] == ("rm", "--force")


def test_patch_provider_content_only_counts_separate_request():
    class Client:
        def chat(self, messages, **kwargs):
            assert kwargs["temperature"] == 0.0 and "tools" not in kwargs
            return {"choices": [{"message": {"content": DIFF}}], "usage": {"completion_tokens": 10}}
    provider = QwenPatchProposalProvider(Client())
    assert provider.propose({"candidate": "C1"}) == DIFF
    assert provider.requests == 1 and provider.usage["completion_tokens"] == 10


def test_patch_cli_is_static(project, tmp_path, capsys):
    document = tmp_path / "proposal.json"
    document.write_text(json.dumps(proposal().as_dict()))
    assert cli.main(["predictive-patch-check", "--project", str(project), "--proposal", str(document), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "NOT_TESTED" and value["execution_authorized"] is False


def test_artifact_compaction_preserves_verified_bytes_and_metrics(tmp_path):
    path = tmp_path / "report.json"
    raw = json.dumps({"cases": 1, "results": [{"id": "x", "metrics": {"hit": True}}], "metrics": {"score": 0.9}}).encode()
    path.write_bytes(raw)
    archive = tmp_path / "archive"
    metadata = compact_report(path, archive, source_commit="abc")
    assert gzip.decompress((archive / metadata["archive_name"]).read_bytes()) == raw
    assert metadata["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    compact = json.loads(path.read_text())
    assert compact["metrics"]["score"] == 0.9 and "results" not in compact
    assert compact_report(path, archive, source_commit="abc") == metadata


def test_historical_seal_is_line_ending_independent_and_ignores_directory_marker(tmp_path):
    from tern.orchestrator.predictive.evaluation import predictive_corpus_hash
    (tmp_path / "cases").mkdir()
    fixture = tmp_path / "projects" / "one"
    fixture.mkdir(parents=True)
    (tmp_path / "cases" / "one.jsonl").write_text(json.dumps({
        "id": "one", "split": "development", "project_fixture": "one",
    }) + "\n")
    source = fixture / "module.py"
    source.write_bytes(b"x = 1\n")
    original = predictive_corpus_hash(tmp_path, split="development")
    source.write_bytes(b"x = 1\r\n")
    (fixture / ".gitkeep").touch()
    assert predictive_corpus_hash(tmp_path, split="development") == original
    source.write_bytes(b"x = 2\n")
    assert predictive_corpus_hash(tmp_path, split="development") != original


@pytest.mark.skipif(os.environ.get("RUN_PREDICTIVE_DOCKER_TESTS") != "true", reason="Docker integration is opt-in")
def test_real_docker_selftest():
    provider = detect_sandbox_provider()
    if provider.capabilities().provider == "unavailable":
        pytest.skip(provider.capabilities().reason)
    assert provider.capabilities().ready, provider.capabilities().reason
