from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tern.orchestrator.decision_policy import (
    AgentDecisionPolicy,
    Intent,
    SideEffect,
    constraint_violation_for_tool,
)
from tern.orchestrator.intent_semantics import Constraint, FollowupType, SpeechAct
from tern.orchestrator.routing_eval import evaluate_cases, load_cases


def decide(text: str, **context):
    return AgentDecisionPolicy().decide(text, fixture_context=context)


def constraints(value) -> set[Constraint]:
    return set(value.intent_frame.constraints)


@pytest.mark.parametrize(
    "text",
    [
        "delegue uma tarefa para o deepseek, peça para ele ver como está o firebase do site da simpleenglish",
        "delegue ao deepseek: ver como está o firebase",
        "peça ao deepseek para revisar como está a arquitetura",
        "encaminhe ao deepseek a revisão do firebase",
    ],
)
def test_status_wording_inside_delegated_task_stays_a_delegation(text: str):
    """"como está" describing the delegated content must not become a status query."""
    value = decide(text)
    assert value.intent == Intent.DEEPSEEK_DELEGATE
    assert value.tools == ("delegate_to_deepseek",)
    assert value.intent_frame.speech_act == SpeechAct.COMMAND
    assert value.intent_frame.execution_requested is True
    assert value.constraint_violation is None


@pytest.mark.parametrize(
    "text",
    [
        "como funciona a delegação para o deepseek?",
        "quero entender como se cancela um job",
        "por que o codex falhou?",
    ],
)
def test_meta_discussion_about_capabilities_stays_direct_answer(text: str):
    value = decide(text)
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.tools == ()
    assert value.intent_frame.execution_requested is False


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("sem delegar ao deepseek, me explique o firebase", Constraint.FORBID_DEEPSEEK),
        ("sem mandar para o codex, me explique o bug", Constraint.FORBID_CODEX),
    ],
)
def test_verb_inside_prohibition_is_not_an_execution_request(text: str, forbidden: Constraint):
    value = decide(text)
    assert forbidden in constraints(value)
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.intent_frame.execution_requested is False


def test_question_about_cancel_is_not_a_cancel_command():
    question = decide(
        "como eu cancelo o Codex?",
        focused_agent="codex",
        codex_job={"status": "running", "job_id": "job-1"},
    )
    command = decide(
        "cancela o Codex",
        focused_agent="codex",
        codex_job={"status": "running", "job_id": "job-1"},
    )
    assert question.intent == Intent.ANSWER_DIRECTLY
    assert question.intent_frame.speech_act == SpeechAct.EXPLANATION_REQUEST
    assert question.intent_frame.execution_requested is False
    assert command.intent == Intent.CODEX_CANCEL
    assert command.intent_frame.execution_requested is True


def test_negation_scope_keeps_the_positive_status_clause():
    value = decide(
        "não cancela, só vê se terminou",
        focused_agent="codex",
        codex_job={"status": "running", "job_id": "job-1"},
    )
    assert value.intent == Intent.CODEX_STATUS
    assert value.tools == ("get_codex_job_status",)
    assert Constraint.FORBID_CANCEL in constraints(value)


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ("sem Codex, responde você", Constraint.FORBID_CODEX),
        ("não consulta o DeepSeek, quero sua opinião", Constraint.FORBID_DEEPSEEK),
        ("use só leitura para verificar isso", Constraint.READ_ONLY),
        ("revise a sessão, não crie outro turn", Constraint.FORBID_NEW_TURN),
    ],
)
def test_explicit_constraints_are_represented(text, forbidden):
    value = decide(text, active_project="tern", content_available=True)
    assert forbidden in constraints(value)


def test_contradictory_constraints_require_clarification():
    assert decide("não usa o Codex, manda o Codex corrigir", active_project="tern").intent == Intent.CLARIFY
    assert decide("não pergunta ao DeepSeek, pergunta ao DeepSeek", active_project="tern").intent == Intent.CLARIFY


@pytest.mark.parametrize(
    "text",
    [
        "o DeepSeek é melhor pra isso?",
        "o DeepSeek conseguiria analisar isso?",
        "como funciona delegate_to_codex?",
        "quero saber como cancelar o Codex",
    ],
)
def test_agent_or_action_mention_is_not_execution(text):
    value = decide(text, active_project="tern", focused_agent="codex", codex_job={"status": "running"})
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.tools == ()
    assert value.intent_frame.execution_requested is False


def test_pronouns_resolve_by_entity_type_and_verb_compatibility():
    job = decide(
        "ele terminou?",
        focused_agent="deepseek",
        codex_job={"status": "running", "job_id": "job-1"},
        focused_session="ds-1",
    )
    response = decide(
        "o que ele falou?",
        focused_agent="deepseek",
        focused_session="ds-1",
    )
    file_value = decide("abre ele", focused_file=r"D:\tern\config.py")
    assert job.resolved_reference.type == "codex_job"
    assert response.resolved_reference.type == "deepseek_session"
    assert file_value.resolved_reference.type == "file"


def test_reference_ambiguity_is_not_guessed():
    value = decide(
        "ele terminou?",
        recent_entities=[
            {"type": "codex_job", "id": "job-1", "turn_index": 5},
            {"type": "generation", "id": "ds-gen", "turn_index": 5},
        ],
        turn_index=5,
    )
    assert value.intent == Intent.CLARIFY
    assert value.resolved_reference.ambiguous


def test_focus_decay_and_stack_prefer_compatible_recent_entity():
    value = decide(
        "abre ele",
        focused_file=r"D:\tern\old.py",
        recent_entities=[
            {"type": "file", "id": r"D:\tern\old.py", "turn_index": 1},
            {"type": "file", "id": r"D:\tern\new.py", "turn_index": 8},
        ],
        turn_index=8,
    )
    assert value.intent == Intent.LOCAL_READ
    assert value.target == r"D:\tern\new.py"


def test_correction_the_other_uses_focus_stack_without_executing():
    value = decide(
        "não, o outro",
        focused_file=r"D:\tern\new.py",
        recent_entities=[
            {"type": "file", "id": r"D:\tern\old.py", "turn_index": 6},
            {"type": "file", "id": r"D:\tern\new.py", "turn_index": 7},
        ],
        turn_index=7,
    )
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.intent_frame.speech_act == SpeechAct.CORRECTION
    assert value.resolved_reference.id == r"D:\tern\old.py"
    assert value.tools == ()


def test_compound_actions_are_ordered_and_constraints_remove_forbidden_step():
    ordered = decide(
        "pergunta ao DeepSeek uma solução e depois manda o Codex implementar",
        active_project="tern",
    )
    limited = decide(
        "vê com o DeepSeek mas não manda nada pro Codex",
        active_project="tern",
    )
    assert ordered.tools == ("delegate_to_deepseek", "delegate_to_codex")
    assert Constraint.ORDERED in constraints(ordered)
    assert limited.tools == ("delegate_to_deepseek",)
    assert Constraint.FORBID_CODEX in constraints(limited)


def test_followup_types_distinguish_status_steer_reference_and_correction():
    running = {"focused_agent": "codex", "codex_job": {"status": "running", "job_id": "job-1"}}
    assert decide("e aí terminou?", **running).intent_frame.followup_type == FollowupType.STATUS_FOLLOWUP
    assert decide("e os warnings?", **running).intent_frame.followup_type == FollowupType.MODIFICATION
    assert decide("e a resposta anterior?", focused_agent="deepseek", focused_session="ds-1").intent_frame.followup_type == FollowupType.REFERENCE_FOLLOWUP
    assert decide("quis dizer o DeepSeek", focused_agent="codex").intent_frame.followup_type == FollowupType.CORRECTION


def test_available_tool_result_is_reused():
    value = decide(
        "o que você achou do que ele fez?",
        focused_agent="codex",
        content_available=True,
        recent_tools=["review_codex_session"],
    )
    assert value.intent == Intent.ANSWER_DIRECTLY
    assert value.tools == ()
    assert value.reason_code == "tool_result_already_available"


@pytest.mark.parametrize(
    ("text", "tool"),
    [
        ("não usa o Codex, responde você", "delegate_to_codex"),
        ("não consulta o DeepSeek, quero sua opinião", "delegate_to_deepseek"),
        ("não cancela, só vê se terminou", "cancel_codex_job"),
        ("revise sem criar outro turn", "delegate_to_codex"),
    ],
)
def test_constraint_validation_blocks_incompatible_tool(text, tool):
    value = decide(text, active_project="tern", focused_agent="codex", codex_job={"status": "running"})
    assert constraint_violation_for_tool(tool, value.intent_frame)


def test_execution_not_requested_blocks_side_effects_but_not_reads():
    value = decide("como eu cancelo o Codex?", focused_agent="codex")
    assert constraint_violation_for_tool("cancel_codex_job", value.intent_frame) == "EXECUTION_NOT_REQUESTED"
    assert constraint_violation_for_tool("review_codex_session", value.intent_frame) is None
    assert SideEffect.READ_ONLY.value in [SideEffect.READ_ONLY.value]


def test_known_routing_corpora_are_regression_gates():
    root = Path(__file__).parent / "data"
    expected = {
        "agent_routing_cases.jsonl": 100,
        "agent_routing_test_v2.jsonl": 40,
        "agent_routing_semantic_regression.jsonl": 64,
    }
    for name, count in expected.items():
        cases = load_cases(root / name)
        policy = AgentDecisionPolicy()
        report = evaluate_cases(
            cases,
            lambda text, context: policy.decide(text, fixture_context=context).as_dict(),
        )
        assert report["cases"] == count
        assert report["passed"] == count
        assert report["forbidden_tool_calls"] == 0
        assert report["new_turn_violations"] == 0
        assert report["tool_loop_violations"] == 0


def test_semantic_regression_has_required_size_and_categories():
    cases = load_cases(Path(__file__).parent / "data" / "agent_routing_semantic_regression.jsonl")
    categories = Counter(case["category"] for case in cases)
    assert 60 <= len(cases) <= 100
    assert set(categories) == {
        "negation",
        "meta_discussion",
        "references",
        "followups",
        "corrections",
        "agent_switch",
        "compound",
        "constraints",
    }


def test_semantic_properties_hold_for_entire_known_corpus():
    cases = load_cases(Path(__file__).parent / "data" / "agent_routing_semantic_regression.jsonl")
    policy = AgentDecisionPolicy()
    for case in cases:
        value = policy.decide(case["input"], fixture_context=case["context"])
        frame = value.intent_frame
        for tool in value.tools:
            assert constraint_violation_for_tool(tool, frame) is None, case["id"]
        if Constraint.FORBID_DEEPSEEK in frame.constraints:
            assert "delegate_to_deepseek" not in value.tools, case["id"]
        if Constraint.FORBID_NEW_TURN in frame.constraints:
            assert "delegate_to_codex" not in value.tools, case["id"]
        if not frame.execution_requested:
            assert all(effect == SideEffect.READ_ONLY for effect in value.side_effects), case["id"]
