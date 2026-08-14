from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tern.orchestrator.autonomy_eval import (
    AutonomyEvaluator,
    diagnostic_baseline,
    load_cases,
    project_understanding_metrics,
)
from tern.orchestrator.autonomy_foundation import (
    Agent,
    Capability,
    CapabilityProfileBuilder,
    EligibilityEngine,
    RiskLevel,
    TaskRequirementAnalyzer,
    TaskRequirements,
    VerificationExpectation,
    propose_agent_selection,
    task_requirement_json_schema,
    verify_facts,
)
from tern.orchestrator.project_intelligence import (
    ExplorationBudget,
    ProjectFileSelector,
    ProjectSnapshotBuilder,
)


CORPUS = Path(__file__).parent / "data" / "autonomy_foundation_diagnostic.jsonl"


def _requirements(*capabilities: Capability) -> TaskRequirements:
    return TaskRequirements(
        frozenset(capabilities),
        Capability.MUTATION in capabilities,
        Capability.MUTATION not in capabilities,
        "test",
        risk_level=RiskLevel.MEDIUM,
    )


def test_capability_profiles_are_derived_and_availability_is_separate():
    baseline = diagnostic_baseline(deepseek_available=False)
    deepseek = baseline.profiles[Agent.DEEPSEEK]
    assert Capability.CODE_ANALYSIS in deepseek.capabilities
    assert Capability.FILESYSTEM_READ not in deepseek.capabilities
    assert not baseline.availability[Agent.DEEPSEEK].available
    assert baseline.availability[Agent.DEEPSEEK].reason_code == "DEEPSEEK_NOT_CONFIGURED"
    codex = baseline.profiles[Agent.CODEX]
    assert {
        Capability.REPOSITORY_WRITE,
        Capability.TEST_EXECUTION,
        Capability.PERSISTENT_SESSION,
        Capability.LONG_RUNNING_JOB,
    } <= codex.capabilities


def test_task_requirement_schema_has_no_agent_selection_fields():
    properties = task_requirement_json_schema()["json_schema"]["schema"]["properties"]
    assert "requested_agent" not in properties
    assert "selected_agent" not in properties
    assert "proposed_agent" not in properties


def test_single_multiple_and_no_eligible_agents():
    baseline = diagnostic_baseline()
    engine = EligibilityEngine()
    single = engine.evaluate(
        _requirements(Capability.TEST_EXECUTION), baseline.profiles, baseline.availability
    )
    assert propose_agent_selection(single, requested_agent=None).proposed_agent is Agent.CODEX
    multiple = engine.evaluate(
        _requirements(Capability.GENERAL_REASONING), baseline.profiles, baseline.availability
    )
    assert propose_agent_selection(multiple, requested_agent=None).reason_code == "MULTIPLE_ELIGIBLE_AGENTS"
    none = engine.evaluate(
        _requirements(Capability.TEST_EXECUTION, Capability.WEB_ACCESS),
        baseline.profiles,
        baseline.availability,
    )
    assert propose_agent_selection(none, requested_agent=None).reason_code == "NO_ELIGIBLE_AGENT"


def test_requested_agent_beats_policy_without_fallback():
    baseline = diagnostic_baseline()
    evaluations = EligibilityEngine().evaluate(
        _requirements(Capability.TEST_EXECUTION), baseline.profiles, baseline.availability
    )
    proposal = propose_agent_selection(evaluations, requested_agent=Agent.DEEPSEEK)
    assert proposal.selected_agent is Agent.DEEPSEEK
    assert proposal.proposed_agent is None
    assert proposal.reason_code == "REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS"
    assert not proposal.execution_authorized


def test_deepseek_unavailable_is_not_removed_from_capability_eligibility():
    baseline = diagnostic_baseline(deepseek_available=False)
    evaluations = EligibilityEngine().evaluate(
        _requirements(Capability.GENERAL_REASONING), baseline.profiles, baseline.availability
    )
    assert evaluations[Agent.DEEPSEEK].eligible
    assert not evaluations[Agent.DEEPSEEK].executable_now


class _Client:
    def __init__(self, content: dict):
        self.content = content
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(self.content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


class _FailingClient:
    def chat(self, *_args, **_kwargs):
        raise RuntimeError("incompatible runtime")


def test_requirement_analyzer_asks_for_requirements_before_agents():
    value = {
        "capabilities": ["repository_read"],
        "mutation_required": False,
        "read_only_required": True,
        "target_scope": "repository",
        "risk_level": "low",
        "expected_files": [],
        "forbidden_files": [],
        "tests_requested": [],
        "ambiguity_material": False,
    }
    client = _Client(value)
    result = TaskRequirementAnalyzer(client).analyze("inspecione o projeto")
    assert result.valid and result.first_pass_valid
    schema = client.calls[0][1]["response_format"]
    assert "agent" not in json.dumps(schema).casefold()


def test_requirement_analyzer_reports_runtime_incompatibility_without_fallback():
    result = TaskRequirementAnalyzer(_FailingClient()).analyze("analise")
    assert not result.valid
    assert result.attempts == 2
    assert result.error_code == "RuntimeError"


def test_project_snapshot_detects_changes_symbols_and_reuses_cache(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "main.py").write_text("import json\n\ndef run():\n    return json.dumps({})\n", encoding="utf-8")
    (root / "tests" / "test_main.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname="demo"\nversion="0.1"\n[project.scripts]\ndemo="src.main:run"\n',
        encoding="utf-8",
    )
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    cache = tmp_path / "cache.json"
    budget = ExplorationBudget(max_files_per_analysis=10, max_total_context_bytes=100_000)
    first = ProjectSnapshotBuilder(root, cache_path=cache, budget=budget).build()
    assert "Python" in first.languages
    assert "tests/test_main.py" in first.tests
    main = next(item for item in first.repo_map if item.path == "src/main.py")
    assert main.symbols == ("run",)
    assert main.imports == ("json",)
    second = ProjectSnapshotBuilder(root, cache_path=cache, budget=budget).build()
    assert second.files_reused == len(second.repo_map)


def test_project_snapshot_obeys_exploration_budget(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(6):
        (root / f"m{index}.py").write_text(f"def f{index}():\n    pass\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder(
        root,
        budget=ExplorationBudget(
            max_files_per_analysis=2,
            max_bytes_per_file=100,
            max_total_context_bytes=200,
        ),
    ).build()
    assert snapshot.files_analyzed == 2


def test_compact_snapshot_obeys_serialized_context_budget(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(100):
        (root / f"module_{index}.py").write_text(
            "\n".join(f"def symbol_{item}(): pass" for item in range(30)),
            encoding="utf-8",
        )
    compact = ProjectSnapshotBuilder(root).build().compact(max_context_bytes=8_000)
    assert len(json.dumps(compact, ensure_ascii=False).encode("utf-8")) <= 8_000
    assert compact["truncated"]


def test_verification_uses_external_facts():
    result = verify_facts(
        VerificationExpectation(
            expected_files=frozenset({"src/a.py"}),
            forbidden_files=frozenset({"secrets.env"}),
            tests_requested=("pytest",),
        ),
        actual_files_changed=("src/a.py", "secrets.env"),
        tests_executed=("pytest",),
        test_exit_code=0,
        objective_satisfied=True,
    )
    assert result.status == "failed"
    assert result.scope_violation
    assert result.forbidden_files_touched == ("secrets.env",)


def test_diagnostic_corpus_is_perfect_and_never_executes():
    report = AutonomyEvaluator(diagnostic_baseline()).evaluate(load_cases(CORPUS))
    metrics = report["metrics"]
    assert metrics["eligibility_precision"] == 1.0
    assert metrics["eligibility_recall"] == 1.0
    assert metrics["single_candidate_resolution_accuracy"] == 1.0
    assert metrics["no_execution_from_dry_run_accuracy"] == 1.0
    assert report["execution_count"] == 0


def test_project_understanding_metrics():
    metrics = project_understanding_metrics(
        ["a.py", "b.py"], ["a.py", "b.py", "irrelevant.md"]
    )
    assert metrics == {"relevant_file_recall": 1.0, "irrelevant_file_selection_rate": 1 / 3}


def test_project_file_selector_rejects_paths_outside_snapshot():
    client = _Client({"selected_files": ["not/in/snapshot.py"], "probable_modules": []})
    result = ProjectFileSelector(client).select(
        "inspect",
        {"repo_map": [{"path": "src/main.py"}], "tests": [], "important_files": []},
    )
    assert not result.valid
    assert result.selected_files == ()
