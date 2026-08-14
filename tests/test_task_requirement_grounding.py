from __future__ import annotations

import json

import pytest

from tern.orchestrator.autonomy_eval import diagnostic_baseline
from tern.orchestrator.autonomy_foundation import Agent, Capability, RiskLevel
from tern.orchestrator.grounding_eval import GroundingEvaluator, load_grounding_cases
from tern.orchestrator.task_requirement_grounding import (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
    REQUIREMENT_DIMENSIONS,
    ExplicitTaskEvidenceExtractor,
    GroundedEligibilityEngine,
    GroundedRequirement,
    GroundedRequirementAnalysisResult,
    GroundedTaskRequirementAnalyzer,
    GroundedTaskRequirements,
    RequirementEvidenceSource,
    RequirementValue,
    grounded_requirement,
    merge_grounded_requirements,
)


def requirement(name, value, source, ref="test:evidence"):
    return grounded_requirement(name, value, source, ref)


def grounded(**changes):
    values = {name: GroundedRequirement.unknown(name) for name in REQUIREMENT_DIMENSIONS}
    values.update(changes)
    return GroundedTaskRequirements(values, "test", RiskLevel.LOW, False)


class StaticClient:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "choices": [
                {
                    "message": {"content": json.dumps(self.value)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (RequirementValue.TRUE, "TRUE"),
        (RequirementValue.FALSE, "FALSE"),
        (RequirementValue.UNKNOWN, "UNKNOWN"),
    ],
)
def test_requirement_values_are_explicit(value, expected):
    item = requirement(
        Capability.WEB_ACCESS.value,
        value,
        RequirementEvidenceSource.SEMANTIC_INFERENCE,
    )
    assert item.as_dict()["value"] == expected


def test_explicit_positive_mutation_and_tests_are_preserved():
    result = ExplicitTaskEvidenceExtractor().extract(
        "corrija o código e execute os testes"
    )
    assert result.requirements[MUTATION_REQUIRED].value is RequirementValue.TRUE
    assert (
        result.requirements[MUTATION_REQUIRED].source
        is RequirementEvidenceSource.EXPLICIT_USER
    )
    assert (
        result.requirements[Capability.TEST_EXECUTION.value].value
        is RequirementValue.TRUE
    )
    assert result.tests_requested == ("explicit:test_execution",)


def test_explicit_read_only_is_stronger_than_mutation_wording():
    result = ExplicitTaskEvidenceExtractor().extract(
        "revise este módulo sem modificar nada"
    )
    assert result.requirements[MUTATION_REQUIRED].value is RequirementValue.FALSE
    assert (
        result.requirements[MUTATION_REQUIRED].source
        is RequirementEvidenceSource.EXPLICIT_USER_NEGATION
    )
    assert result.requirements[READ_ONLY_REQUIRED].value is RequirementValue.TRUE
    assert result.requirements[Capability.CODE_EDIT.value].value is RequirementValue.FALSE


def test_contradictory_explicit_mutation_and_read_only_produce_conflict():
    result = ExplicitTaskEvidenceExtractor().extract(
        "corrija este módulo, mas sem modificar nada"
    )
    assert result.requirements[MUTATION_REQUIRED].value is RequirementValue.CONFLICT


def test_explicit_no_tests_and_no_web_are_preserved():
    result = ExplicitTaskEvidenceExtractor().extract(
        "corrija sem rodar testes e não use a web"
    )
    assert result.requirements[Capability.TEST_EXECUTION.value].value is RequirementValue.FALSE
    assert result.requirements[Capability.WEB_ACCESS.value].value is RequirementValue.FALSE
    assert set(result.prohibitions) >= {"test_execution", "web_access"}


def test_explicit_web_is_preserved():
    result = ExplicitTaskEvidenceExtractor().extract(
        "pesquise a documentação oficial e compare"
    )
    item = result.requirements[Capability.WEB_ACCESS.value]
    assert item.value is RequirementValue.TRUE
    assert item.source is RequirementEvidenceSource.EXPLICIT_USER


def test_explicit_scope_and_commit_prohibitions_are_retained():
    result = ExplicitTaskEvidenceExtractor().extract(
        "corrija, não altere a API pública e não faça commit"
    )
    assert set(result.prohibitions) >= {"public_api_change", "commit"}
    assert result.forbidden_files == ("scope:public_api",)


@pytest.mark.parametrize(
    ("text", "agent"),
    [
        ("use Codex para corrigir", Agent.CODEX),
        ("use DeepSeek para analisar", Agent.DEEPSEEK),
    ],
)
def test_requested_agent_binding_and_origin_are_immutable(text, agent):
    result = ExplicitTaskEvidenceExtractor().extract(text)
    assert result.requested_agent is agent
    assert result.requested_agent_source is RequirementEvidenceSource.REQUESTED_AGENT
    assert result.requested_agent_evidence_ref == "binding:executor_clause"


def test_project_fact_evidence_uses_bounded_snapshot_not_file_contents():
    snapshot = {
        "repo_map": [
            {"path": "tern/orchestrator/autonomy_foundation.py", "file_type": "Python"}
        ]
    }
    result = ExplicitTaskEvidenceExtractor().extract(
        "revise tern/orchestrator/autonomy_foundation.py sem modificar nada",
        project_snapshot=snapshot,
    )
    item = result.requirements[Capability.REPOSITORY_READ.value]
    assert item.value is RequirementValue.TRUE
    assert item.source is RequirementEvidenceSource.PROJECT_FACT
    assert item.evidence_refs == ("project:target_path_exists",)


def test_existing_structured_search_operation_grounds_repository_inspection():
    result = ExplicitTaskEvidenceExtractor().extract(
        "descubra onde o routing é implementado neste projeto",
        project_snapshot={
            "project_path": r"C:\workspace\JARVIS",
            "repo_map": [],
        },
    )
    assert result.target_scope == "repository"
    assert result.requirements[Capability.REPOSITORY_READ.value].value is RequirementValue.TRUE
    assert result.requirements[Capability.CODE_ANALYSIS.value].value is RequirementValue.TRUE
    assert result.requirements[MUTATION_REQUIRED].value is RequirementValue.FALSE
    assert result.requirements[READ_ONLY_REQUIRED].value is RequirementValue.TRUE


def test_strong_evidence_beats_semantic_inference():
    explicit = requirement(
        Capability.WEB_ACCESS.value,
        RequirementValue.FALSE,
        RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
        "constraint:forbid_web",
    )
    inferred = requirement(
        Capability.WEB_ACCESS.value,
        RequirementValue.TRUE,
        RequirementEvidenceSource.SEMANTIC_INFERENCE,
        "model:resolved_dimension",
    )
    assert merge_grounded_requirements(explicit, inferred) == explicit


def test_contradictory_strong_evidence_produces_conflict():
    positive = requirement(
        MUTATION_REQUIRED,
        RequirementValue.TRUE,
        RequirementEvidenceSource.EXPLICIT_USER,
        "action:mutation",
    )
    negative = requirement(
        MUTATION_REQUIRED,
        RequirementValue.FALSE,
        RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
        "constraint:forbid_mutation",
    )
    merged = merge_grounded_requirements(positive, negative)
    assert merged.value is RequirementValue.CONFLICT
    assert merged.source is RequirementEvidenceSource.CONFLICTING_EVIDENCE
    assert set(merged.evidence_refs) == {"action:mutation", "constraint:forbid_mutation"}


def test_model_resolves_only_unresolved_fields_with_one_call():
    client = StaticClient(
        {
            "resolved": [
                {"name": Capability.CODE_ANALYSIS.value, "value": "TRUE"}
            ],
            "target_scope": "concept",
            "risk_level": "low",
            "ambiguity_material": False,
        }
    )
    result = GroundedTaskRequirementAnalyzer(client).analyze(
        "explique como duas abordagens funcionam"
    )
    assert result.valid is True
    assert result.inference_count == 1
    assert len(client.calls) == 1
    inferred = result.requirements.requirements[Capability.CODE_ANALYSIS.value]
    assert inferred.value is RequirementValue.TRUE
    assert inferred.source is RequirementEvidenceSource.SEMANTIC_INFERENCE
    assert result.requirements.requirements[Capability.WEB_ACCESS.value].value is RequirementValue.UNKNOWN
    payload = json.loads(client.calls[0][0][1]["content"])
    assert MUTATION_REQUIRED in payload["known_evidence"]["requirements"]
    assert MUTATION_REQUIRED not in payload["unresolved_dimensions"]
    assert Capability.SEMANTIC_INTERPRETATION.value not in payload["unresolved_dimensions"]
    assert Capability.TASK_REQUIREMENT_EXTRACTION.value not in payload["unresolved_dimensions"]
    assert Capability.LONG_RUNNING_JOB.value not in payload["unresolved_dimensions"]
    assert Capability.PERSISTENT_SESSION.value not in payload["unresolved_dimensions"]


def test_omitted_model_fields_remain_insufficient_evidence():
    client = StaticClient(
        {
            "resolved": [],
            "target_scope": "unknown",
            "risk_level": "medium",
            "ambiguity_material": True,
        }
    )
    result = GroundedTaskRequirementAnalyzer(client).analyze("melhore isso")
    web = result.requirements.requirements[Capability.WEB_ACCESS.value]
    assert web.value is RequirementValue.UNKNOWN
    assert web.source is RequirementEvidenceSource.INSUFFICIENT_EVIDENCE


def test_semantic_inference_cannot_invent_repository_access_without_scope():
    client = StaticClient(
        {
            "resolved": [
                {"name": Capability.REPOSITORY_READ.value, "value": "TRUE"}
            ],
            "target_scope": "repository",
            "risk_level": "low",
            "ambiguity_material": False,
        }
    )
    result = GroundedTaskRequirementAnalyzer(client).analyze(
        "explique como este conceito funciona"
    )
    repository_read = result.requirements.requirements[
        Capability.REPOSITORY_READ.value
    ]
    assert repository_read.value is RequirementValue.UNKNOWN
    assert repository_read.source is RequirementEvidenceSource.INSUFFICIENT_EVIDENCE


def test_unknown_does_not_eliminate_agent_or_authorize_mutation():
    baseline = diagnostic_baseline()
    evaluations = GroundedEligibilityEngine().evaluate(
        grounded(), baseline.profiles, baseline.availability
    )
    assert set(evaluations) == {Agent.LOCAL, Agent.CODEX, Agent.DEEPSEEK}
    assert all(item.eligible for item in evaluations.values())
    assert all(not item.executable_now for item in evaluations.values())
    assert all(not item.execution_safe for item in evaluations.values())
    assert all(item.authorized_capabilities == () for item in evaluations.values())


def test_true_requirement_controls_eligibility_while_unknown_does_not():
    baseline = diagnostic_baseline()
    requirements = grounded(
        **{
            MUTATION_REQUIRED: requirement(
                MUTATION_REQUIRED,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
            ),
            Capability.TEST_EXECUTION.value: requirement(
                Capability.TEST_EXECUTION.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
            ),
        }
    )
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, baseline.availability
    )
    assert evaluations[Agent.CODEX].eligible is True
    assert evaluations[Agent.LOCAL].eligible is False
    assert evaluations[Agent.DEEPSEEK].eligible is False


def test_multiple_eligible_agents_are_preserved_without_ranking():
    baseline = diagnostic_baseline()
    requirements = grounded(
        **{
            MUTATION_REQUIRED: requirement(
                MUTATION_REQUIRED,
                RequirementValue.FALSE,
                RequirementEvidenceSource.TASK_IMPLICATION,
            ),
            Capability.GENERAL_REASONING.value: requirement(
                Capability.GENERAL_REASONING.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.SEMANTIC_INFERENCE,
            ),
        }
    )
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, baseline.availability
    )
    assert {agent for agent, item in evaluations.items() if item.eligible} == set(Agent)


def test_no_eligible_agent_for_combined_web_and_test_requirement():
    baseline = diagnostic_baseline()
    requirements = grounded(
        **{
            MUTATION_REQUIRED: requirement(
                MUTATION_REQUIRED,
                RequirementValue.FALSE,
                RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
            ),
            Capability.WEB_ACCESS.value: requirement(
                Capability.WEB_ACCESS.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
            ),
            Capability.TEST_EXECUTION.value: requirement(
                Capability.TEST_EXECUTION.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
            ),
        }
    )
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, baseline.availability
    )
    assert not any(item.eligible for item in evaluations.values())


def test_mutation_false_conflicting_with_write_blocks_execution():
    baseline = diagnostic_baseline()
    requirements = grounded(
        **{
            MUTATION_REQUIRED: requirement(
                MUTATION_REQUIRED,
                RequirementValue.FALSE,
                RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
            ),
            Capability.REPOSITORY_WRITE.value: requirement(
                Capability.REPOSITORY_WRITE.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.SEMANTIC_INFERENCE,
            ),
        }
    )
    evaluations = GroundedEligibilityEngine().evaluate(
        requirements, baseline.profiles, baseline.availability
    )
    assert all(not item.execution_safe for item in evaluations.values())
    assert all(
        Capability.REPOSITORY_WRITE.value in item.conflict_requirements
        for item in evaluations.values()
    )


def test_evidence_refs_do_not_persist_user_prompt():
    secret = "corrija o código token-super-secreto-123"
    result = ExplicitTaskEvidenceExtractor().extract(secret)
    rendered = json.dumps(result.as_dict(), ensure_ascii=False)
    assert secret not in rendered
    assert "token-super-secreto-123" not in rendered


def test_diagnostic_corpus_deterministic_evidence_matches_labels():
    cases = load_grounding_cases(
        "tests/data/task_requirement_grounding_diagnostic.jsonl"
    )
    snapshot = {
        "repo_map": [
            {"path": "tern/orchestrator/autonomy_foundation.py", "file_type": "Python"}
        ]
    }
    extractor = ExplicitTaskEvidenceExtractor()
    model_owned = {
        RequirementEvidenceSource.SEMANTIC_INFERENCE,
        RequirementEvidenceSource.INSUFFICIENT_EVIDENCE,
    }
    for case in cases:
        evidence = extractor.extract(case.input, project_snapshot=snapshot)
        for name, expected in case.expected.items():
            if expected.source in model_owned:
                continue
            actual = evidence.requirements[name]
            assert (actual.value, actual.source) == (
                expected.value,
                expected.source,
            ), (case.id, name)


def test_perfect_grounded_evaluator_scores_one_and_never_executes():
    cases = load_grounding_cases(
        "tests/data/task_requirement_grounding_diagnostic.jsonl"
    )
    analyses = []
    for case in cases:
        values = {
            name: GroundedRequirement(
                name,
                expected.value,
                expected.source,
                ("fixture:expected",),
                requirement(name, expected.value, expected.source).safety_class,
            )
            for name, expected in case.expected.items()
        }
        requirements = GroundedTaskRequirements(
            values,
            "fixture",
            RiskLevel.LOW,
            case.expected_ambiguity,
            prohibitions=tuple(case.expected_prohibitions),
            requested_agent=case.expected_requested_agent,
            requested_agent_source=(
                RequirementEvidenceSource.REQUESTED_AGENT
                if case.expected_requested_agent
                else None
            ),
            requested_agent_evidence_ref=(
                "fixture:binding" if case.expected_requested_agent else None
            ),
        )
        analyses.append(
            GroundedRequirementAnalysisResult(
                requirements,
                True,
                True,
                1,
                1,
                1.0,
                "stop",
                1,
                1,
            )
        )
    report = GroundingEvaluator().evaluate_variant(
        cases,
        analyses,
        variant="grounded",
    )
    metrics = report["metrics"]
    assert metrics["macro_f1"] == 1.0
    assert metrics["provenance_accuracy"] == 1.0
    assert metrics["explicit_constraint_retention"] == 1.0
    assert metrics["explicit_prohibition_retention"] == 1.0
    assert metrics["false_mutation_authorization"] == 0
    assert report["execution_count"] == 0
