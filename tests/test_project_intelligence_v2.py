from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tern.orchestrator.project_intelligence_v2 import (
    CandidateBudget,
    EvidenceStrength,
    ProjectCandidateGenerator,
    ProjectCandidateRanker,
    ProjectIndexBuilderV2,
    RelevantFileEvidenceSource,
    SymbolKind,
)
from tern.orchestrator.project_intelligence_eval_v2 import (
    ObservedSelection,
    ProjectIntelligenceCase,
    evaluate_project_intelligence,
    load_project_intelligence_cases,
)
from tern.orchestrator.security import AccessDenied, PathPolicy


def write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def git_repo(tmp_path: Path, name: str = "repo") -> Path:
    root = tmp_path / name
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def commit_all(root: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def sources(result, path: str) -> set[RelevantFileEvidenceSource]:
    candidate = next(item for item in result.candidates if item.path == path)
    return {item.source for item in candidate.evidences}


def basic_repo(tmp_path: Path) -> Path:
    root = git_repo(tmp_path)
    write(
        root,
        "pkg/service.py",
        "class SessionResolver:\n"
        "    def resolve_session(self, value):\n"
        "        return value\n",
    )
    write(root, "pkg/caller.py", "from .service import SessionResolver\n")
    write(
        root,
        "tests/test_service.py",
        "from pkg.service import SessionResolver\n\n"
        "def test_session_reuse():\n"
        "    assert SessionResolver().resolve_session('x') == 'x'\n",
    )
    write(root, "pyproject.toml", '[project.scripts]\ndemo="pkg.caller:main"\n')
    commit_all(root)
    return root


def test_python_ast_indexes_qualified_symbols_lines_and_constants(tmp_path):
    root = basic_repo(tmp_path)
    snapshot = ProjectIndexBuilderV2(root).build()
    service = snapshot.file_index["pkg/service.py"]
    indexed = {item.qualified_name: item for item in service.symbols}
    assert indexed["SessionResolver"].kind is SymbolKind.CLASS
    assert indexed["SessionResolver.resolve_session"].kind is SymbolKind.METHOD
    assert indexed["SessionResolver.resolve_session"].line == 2


def test_explicit_file_target_is_hard_and_preserved(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    result = ProjectCandidateGenerator().generate("revise service.py", snapshot)
    assert result.selected_files[0] == "pkg/service.py"
    candidate = result.selected[0]
    assert candidate.hard
    assert RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE in sources(
        result, "pkg/service.py"
    )


def test_explicit_symbol_resolves_definition_with_provenance(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    result = ProjectCandidateGenerator().generate(
        "corrija SessionResolver.resolve_session", snapshot
    )
    assert "pkg/service.py" in result.selected_files
    evidence = sources(result, "pkg/service.py")
    assert RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE in evidence
    assert RelevantFileEvidenceSource.SYMBOL_DEFINITION in evidence


def test_duplicate_symbol_names_preserve_all_definitions(tmp_path):
    root = basic_repo(tmp_path)
    write(root, "pkg/other.py", "def duplicate_symbol():\n    return 2\n")
    write(root, "pkg/service.py", "def duplicate_symbol():\n    return 1\n")
    snapshot = ProjectIndexBuilderV2(root).build()
    result = ProjectCandidateGenerator().generate("corrija duplicate_symbol", snapshot)
    assert {"pkg/service.py", "pkg/other.py"}.issubset(result.selected_files)
    assert all(
        next(item for item in result.candidates if item.path == path).hard
        for path in ("pkg/service.py", "pkg/other.py")
    )


def test_import_and_reverse_import_graph_resolve_relative_modules(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    assert snapshot.import_graph["pkg/caller.py"] == ("pkg/service.py",)
    assert "pkg/caller.py" in snapshot.reverse_import_graph["pkg/service.py"]
    result = ProjectCandidateGenerator().generate("revise service.py", snapshot)
    assert RelevantFileEvidenceSource.REVERSE_IMPORT in sources(
        result, "pkg/caller.py"
    )


def test_test_relationship_requires_structural_support(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    relation = next(
        item
        for item in snapshot.test_relationships
        if item.production_file == "pkg/service.py"
        and item.test_file == "tests/test_service.py"
    )
    assert relation.strength is EvidenceStrength.STRONG
    assert "direct_import" in relation.evidence
    result = ProjectCandidateGenerator().generate(
        "execute testes de SessionResolver.resolve_session",
        snapshot,
        requirements={"test_execution": "TRUE"},
    )
    assert RelevantFileEvidenceSource.TEST_RELATIONSHIP in sources(
        result, "tests/test_service.py"
    )


def test_traceback_path_and_line_are_hard_evidence(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    result = ProjectCandidateGenerator().generate(
        'Traceback: File "pkg/service.py", line 2, in resolve_session', snapshot
    )
    assert result.selected_files[0] == "pkg/service.py"
    assert RelevantFileEvidenceSource.TRACEBACK_REFERENCE in sources(
        result, "pkg/service.py"
    )


def test_known_traceback_file_from_runtime_is_preserved(tmp_path):
    root = basic_repo(tmp_path)
    snapshot = ProjectIndexBuilderV2(root).build(
        known_traceback_files=("pkg/service.py",)
    )
    result = ProjectCandidateGenerator().generate("analise a falha", snapshot)
    assert result.selected_files[0] == "pkg/service.py"
    assert next(item for item in result.selected if item.path == "pkg/service.py").hard


def test_absolute_traceback_from_other_project_is_rejected(tmp_path):
    root = basic_repo(tmp_path)
    other = tmp_path / "other" / "service.py"
    other.parent.mkdir()
    other.write_text("raise RuntimeError\n", encoding="utf-8")
    snapshot = ProjectIndexBuilderV2(root).build()
    result = ProjectCandidateGenerator().generate(
        f'Traceback: File "{other}", line 1, in x', snapshot
    )
    assert RelevantFileEvidenceSource.TRACEBACK_REFERENCE not in {
        evidence.source
        for candidate in result.candidates
        for evidence in candidate.evidences
    }


def test_git_modified_file_is_supporting_not_global_selection(tmp_path):
    root = basic_repo(tmp_path)
    write(root, "pkg/service.py", "class SessionResolver:\n    changed = True\n")
    write(root, "unrelated.py", "changed = True\n")
    snapshot = ProjectIndexBuilderV2(root).build()
    result = ProjectCandidateGenerator().generate("revise service.py", snapshot)
    assert RelevantFileEvidenceSource.GIT_MODIFIED_FILE in sources(
        result, "pkg/service.py"
    )
    assert "unrelated.py" not in result.selected_files


def test_config_and_declared_entrypoint_relationship(tmp_path):
    root = basic_repo(tmp_path)
    write(root, "pkg/caller.py", "def main():\n    return 0\n")
    snapshot = ProjectIndexBuilderV2(root).build()
    assert snapshot.declared_entry_points == ("pkg/caller.py",)
    result = ProjectCandidateGenerator().generate("revise pyproject.toml", snapshot)
    assert "pkg/caller.py" in result.selected_files
    assert RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP in sources(
        result, "pkg/caller.py"
    )


def test_incremental_index_reuses_unchanged_files(tmp_path):
    root = basic_repo(tmp_path)
    cache = tmp_path / "index.json"
    first = ProjectIndexBuilderV2(root, cache_path=cache).build()
    second = ProjectIndexBuilderV2(root, cache_path=cache).build()
    assert first.metrics.files_reused == 0
    assert second.metrics.files_reused == len(second.files)
    assert second.metrics.cache_hit


def test_cache_invalidation_uses_hash_even_if_size_and_mtime_match(tmp_path):
    root = basic_repo(tmp_path)
    cache = tmp_path / "index.json"
    first = ProjectIndexBuilderV2(root, cache_path=cache).build()
    target = root / "pkg" / "service.py"
    original_stat = target.stat()
    raw = target.read_text(encoding="utf-8")
    target.write_text(raw.replace("value", "other"), encoding="utf-8")
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = ProjectIndexBuilderV2(root, cache_path=cache).build()
    assert first.file_index["pkg/service.py"].sha256 != second.file_index[
        "pkg/service.py"
    ].sha256
    assert second.metrics.files_indexed == 1


def test_restart_recovers_persisted_symbol_index(tmp_path):
    root = basic_repo(tmp_path)
    cache = tmp_path / "index.json"
    ProjectIndexBuilderV2(root, cache_path=cache).build()
    recovered = ProjectIndexBuilderV2(root, cache_path=cache).build()
    assert "SessionResolver.resolve_session" in recovered.symbol_index
    assert recovered.metrics.files_reused == len(recovered.files)


def test_cache_isolated_by_project_identity(tmp_path):
    first_root = basic_repo(tmp_path)
    second_root = git_repo(tmp_path, "other")
    write(second_root, "unique.py", "def only_here():\n    return True\n")
    commit_all(second_root)
    cache = tmp_path / "shared-index.json"
    first = ProjectIndexBuilderV2(first_root, cache_path=cache).build()
    second = ProjectIndexBuilderV2(second_root, cache_path=cache).build()
    assert first.project_id != second.project_id
    assert second.metrics.files_reused == 0
    assert set(second.file_index) == {"unique.py"}


def test_path_policy_blocks_project_outside_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    policy = PathPolicy((allowed,))
    with pytest.raises(AccessDenied):
        ProjectIndexBuilderV2(outside, path_policy=policy)


def test_candidate_and_context_budgets_bound_large_repository(tmp_path):
    root = git_repo(tmp_path)
    for index in range(80):
        write(root, f"pkg/feature_{index}.py", f"def feature_{index}():\n    return {index}\n")
    commit_all(root)
    snapshot = ProjectIndexBuilderV2(root).build()
    budget = CandidateBudget(max_candidates=10, max_selected_files=5, max_context_bytes=500)
    result = ProjectCandidateGenerator(budget=budget).generate("revise feature", snapshot)
    assert result.metrics.candidate_count <= 10
    assert result.metrics.selected_count <= 5
    assert result.metrics.context_bytes <= 500
    assert not result.metrics.context_budget_violation


def test_hard_evidence_survives_file_and_context_budget(tmp_path):
    root = git_repo(tmp_path)
    write(root, "a.py", "def same_symbol():\n    return 'a'\n" + "# x\n" * 500)
    write(root, "b.py", "def same_symbol():\n    return 'b'\n" + "# y\n" * 500)
    commit_all(root)
    snapshot = ProjectIndexBuilderV2(root).build()
    result = ProjectCandidateGenerator(
        budget=CandidateBudget(
            max_candidates=1,
            max_selected_files=1,
            max_context_bytes=128,
            max_context_bytes_per_file=64,
        )
    ).generate("corrija same_symbol", snapshot)
    assert set(result.selected_files) == {"a.py", "b.py"}
    assert result.metrics.hard_evidence_dropped == 0
    assert result.metrics.context_bytes <= 128


class RankingClient:
    def __init__(self, selected: list[str]):
        self.selected = selected
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "choices": [
                {"message": {"content": json.dumps({"selected_files": self.selected})}}
            ]
        }


def test_semantic_ranking_cannot_remove_explicit_target(tmp_path):
    root = git_repo(tmp_path)
    write(root, "required.py", "def required_symbol():\n    return 1\n")
    for index in range(8):
        write(root, f"feature_{index}.py", f"def feature_{index}():\n    return {index}\n")
    commit_all(root)
    snapshot = ProjectIndexBuilderV2(root).build()
    client = RankingClient([])
    generator = ProjectCandidateGenerator(
        budget=CandidateBudget(max_selected_files=2),
        ranker=ProjectCandidateRanker(client),
    )
    result = generator.generate("revise required.py e feature", snapshot)
    assert "required.py" in result.selected_files
    assert next(item for item in result.selected if item.path == "required.py").hard
    assert client.calls


def test_mutation_scope_is_separate_from_read_scope(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()

    class Requirements:
        expected_files = ("pkg/service.py",)
        forbidden_files = ("pkg/caller.py",)

    result = ProjectCandidateGenerator().generate(
        "revise service.py", snapshot, requirements=Requirements()
    )
    assert "pkg/caller.py" in result.read_scope
    assert result.allowed_mutation_targets == ("pkg/service.py",)
    assert result.forbidden_mutation_targets == ("pkg/caller.py",)


def test_selection_is_dry_run_and_handoff_does_not_authorize_execution(tmp_path):
    snapshot = ProjectIndexBuilderV2(basic_repo(tmp_path)).build()
    result = ProjectCandidateGenerator().generate("revise service.py", snapshot)
    handoff = result.future_worker_handoff(task="revise", requirements={})
    assert result.dry_run and not result.execution_authorized
    assert handoff["dry_run"] and not handoff["execution_authorized"]


def test_v2_diagnostic_corpus_is_separate_and_has_real_ground_truth():
    path = Path(__file__).parent / "data" / "project_intelligence_v2_diagnostic.jsonl"
    cases = load_project_intelligence_cases(path)
    assert len(cases) == 13
    assert len({item.id for item in cases}) == 13
    assert all(item.required_files and item.expected_evidence for item in cases)


def test_v2_evaluator_separates_required_relevant_optional_and_noise():
    case = ProjectIntelligenceCase(
        "case",
        "fixture",
        "inspect",
        {},
        frozenset({"required.py"}),
        frozenset({"relevant.py"}),
        frozenset({"optional.py"}),
        frozenset({"noise.py"}),
        {
            "required.py": frozenset(
                {RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE}
            )
        },
    )
    observed = ObservedSelection(
        ("required.py", "relevant.py", "optional.py", "noise.py"),
        ("required.py", "relevant.py", "optional.py", "noise.py"),
        {
            "required.py": frozenset(
                {RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE}
            )
        },
        True,
        1.0,
        0.0,
        100,
        25,
        400,
        False,
        0,
        False,
    )
    report = evaluate_project_intelligence(
        [case],
        [observed],
        variant="candidate",
        snapshot_files=frozenset(
            {"required.py", "relevant.py", "optional.py", "noise.py"}
        ),
        index_metrics={},
    )
    assert report["metrics"]["required_file_recall"] == 1.0
    assert report["metrics"]["relevant_file_recall"] == 1.0
    assert report["metrics"]["precision"] == 0.75
    assert report["metrics"]["irrelevant_file_selection_rate"] == 0.25
