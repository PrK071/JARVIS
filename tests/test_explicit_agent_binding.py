from __future__ import annotations

from types import SimpleNamespace

import pytest

from tern.orchestrator.decision_policy import AgentDecisionPolicy, Intent
from tern.orchestrator.explicit_agent_binding import (
    ExplicitAgentBinding,
    availability_for_requested_agent,
    detect_explicit_agent_binding,
)
from tern.orchestrator.explicit_agent_binding_eval import (
    evaluate_explicit_agent_binding_ab,
    load_explicit_agent_cases,
)
from tern.orchestrator.projects import normalize_technical_transcript
from tern.orchestrator.semantic_pass import (
    SemanticPassResult,
    QwenSemanticInterpreter,
    validate_semantic_decision,
)


def semantic_frame(**updates):
    value = {
        "speech_act": "QUESTION",
        "primary_intent": "ANSWER_DIRECTLY",
        "operation": "answer",
        "execution_requested": False,
        "agent": "qwen",
        "target": {"type": "none", "reference": None},
        "constraints": ["READ_ONLY"],
        "followup_type": "NEW_REQUEST",
        "continuation": False,
        "compound_plan": [],
        "ambiguity": {"present": False, "candidates": []},
        "confidence": 0.95,
    }
    value.update(updates)
    return validate_semantic_decision(value)


@pytest.mark.parametrize(
    ("text", "agent"),
    [
        ("mande esta tarefa para o DeepSeek", "deepseek"),
        ("delegue esta análise ao DeepSeek", "deepseek"),
        ("peça ao DeepSeek para revisar isso", "deepseek"),
        ("use o DeepSeek para analisar isso", "deepseek"),
        ("mande esta tarefa para o Codex", "codex"),
        ("delegue esta alteração ao Codex", "codex"),
        ("peça ao Codex para trabalhar nisso", "codex"),
        ("fala pro Codex criar uma landing page", "codex"),
        ("fale para o DeepSeek analisar o erro", "deepseek"),
        ("DeepSeek, revise isso", "deepseek"),
    ],
)
def test_detector_requires_an_explicit_executor_clause(text, agent):
    assert detect_explicit_agent_binding(text).requested_agent == agent


def test_leading_binding_survives_agent_names_inside_a_long_task_body():
    text = """delegue essa tarefa para o Codex
# Experimento de payload
Audite exemplos como \"peça ao DeepSeek para revisar\" e compare Codex e DeepSeek.
Não implemente substituição automática entre Codex e DeepSeek.
"""

    binding = detect_explicit_agent_binding(text)

    assert binding is not None
    assert binding.requested_agent == "codex"


def test_same_line_multi_agent_command_does_not_gain_leading_precedence():
    text = "delegue ao Codex e depois peça ao DeepSeek para revisar o resultado"

    assert detect_explicit_agent_binding(text) is None


def test_named_codex_session_resolves_later_executor_pronoun():
    text = (
        "ja tem uma sessão aberta do codex, fala pra ele criar uma landing "
        "page junto com uma tela de cadastro"
    )

    binding = detect_explicit_agent_binding(text)

    assert binding is not None
    assert binding.requested_agent == "codex"
    assert binding.evidence == "contextual_executor_pronoun"


@pytest.mark.parametrize(
    "text",
    [
        "mande para ele criar a página",
        "Codex está aberto, como eu falo pra ele criar uma página?",
        "compare Codex e DeepSeek, depois fala pra ele criar a página",
    ],
)
def test_unsafe_or_ambiguous_executor_pronouns_do_not_bind(text):
    assert detect_explicit_agent_binding(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "o que é DeepSeek?",
        "DeepSeek é melhor que Qwen?",
        "compare Codex e DeepSeek",
        "o que o Codex faz?",
        "qual a diferença entre Codex e DeepSeek?",
        "DeepSeek está disponível?",
        "como mando uma tarefa para o DeepSeek?",
        "não mande isso para o DeepSeek",
        "não consulte o DeepSeek e consulte o DeepSeek",
        "obtenha uma proposta do DeepSeek e depois peça ao Codex para avaliá-la",
        "mande para ele",
    ],
)
def test_mentions_meta_questions_negation_and_pronouns_do_not_bind(text):
    assert detect_explicit_agent_binding(text) is None


def test_corpus_expectations_match_the_isolated_detector():
    cases = load_explicit_agent_cases()
    assert len(cases) == 24
    for case in cases:
        binding = detect_explicit_agent_binding(
            normalize_technical_transcript(case["input"])
        )
        actual = binding.requested_agent if binding else None
        assert actual == case["expected"].get("requested_agent"), case["id"]


def test_existing_routing_corpora_have_zero_false_bindings():
    from pathlib import Path
    import json

    root = Path(__file__).parent / "data"
    paths = [
        root / "agent_routing_cases.jsonl",
        root / "agent_routing_semantic_regression.jsonl",
        root / "agent_routing_test_v2.jsonl",
        root / "agent_routing_test_v3.jsonl",
    ]
    false_bindings = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            binding = detect_explicit_agent_binding(
                normalize_technical_transcript(case["input"])
            )
            expected_intent = case.get("expected", {}).get("intent")
            if binding and expected_intent not in {
                "CODEX_DELEGATE",
                "DEEPSEEK_DELEGATE",
            }:
                false_bindings.append((path.name, case["id"], case["input"]))

    assert false_bindings == []


@pytest.mark.parametrize(
    ("requested", "wrong_semantic", "intent", "tool"),
    [
        ("deepseek", semantic_frame(), Intent.DEEPSEEK_DELEGATE, "delegate_to_deepseek"),
        (
            "deepseek",
            semantic_frame(
                speech_act="COMMAND",
                primary_intent="CODEX_DELEGATE",
                operation="delegate",
                execution_requested=True,
                agent="codex",
                constraints=[],
            ),
            Intent.DEEPSEEK_DELEGATE,
            "delegate_to_deepseek",
        ),
        (
            "codex",
            semantic_frame(
                speech_act="COMMAND",
                primary_intent="DEEPSEEK_DELEGATE",
                operation="delegate",
                execution_requested=True,
                agent="deepseek",
                constraints=[],
            ),
            Intent.CODEX_DELEGATE,
            "delegate_to_codex",
        ),
    ],
)
def test_binding_precedes_a_wrong_semantic_agent_without_repair(
    requested, wrong_semantic, intent, tool
):
    policy = AgentDecisionPolicy()
    context = policy.build_context(
        fixture_context={
            "active_project": "tern",
            "deepseek": {"enabled": True, "configured": True},
        }
    )
    decision = policy.decide(
        f"mande isso para o {requested}",
        context=context,
        semantic_decision=wrong_semantic,
        explicit_agent_binding=ExplicitAgentBinding(requested),
    )

    assert decision.intent is intent
    assert decision.tools == (tool,)
    assert decision.requested_agent == requested
    assert decision.requested_agent_source == "explicit_user"
    assert decision.semantic_frame == wrong_semantic.as_dict()


def test_invalid_semantic_fallback_does_not_erase_upstream_binding():
    policy = AgentDecisionPolicy()
    decision = policy.decide(
        "mande isso para o DeepSeek",
        fixture_context={
            "active_project": "tern",
            "deepseek": {"enabled": True, "configured": True},
        },
        explicit_agent_binding=ExplicitAgentBinding("deepseek"),
    )
    fallback = policy.safe_fallback_decision(decision)

    assert fallback.intent is Intent.DEEPSEEK_DELEGATE
    assert fallback.requested_agent == "deepseek"
    assert fallback.reason_code == "explicit_agent_binding_semantic_parse_failed"


def test_deepseek_availability_is_operational_state_not_semantic_state():
    binding = ExplicitAgentBinding("deepseek")
    enabled = availability_for_requested_agent(
        binding,
        SimpleNamespace(deepseek_enabled=True, deepseek_configured=True),
        {"delegate_to_deepseek"},
    )
    disabled = availability_for_requested_agent(
        binding,
        SimpleNamespace(deepseek_enabled=False, deepseek_configured=True),
        {"delegate_to_deepseek"},
    )

    assert enabled.tool_available and enabled.execution_allowed
    assert not disabled.tool_available and not disabled.execution_allowed
    assert disabled.reason == "agent_disabled"
    assert binding.requested_agent == "deepseek"


def test_ab_reuses_one_semantic_result_and_binding_fixes_only_explicit_cases():
    def provider(original, _normalized, _context):
        if original == "delegue esta análise ao DeepSeek":
            return SemanticPassResult(
                used=True,
                decision=None,
                latency_ms=2.0,
                parse_valid=False,
                repair_used=True,
                cache_hit=False,
                error="semantic_parse_failed",
            )
        if original == "mande para ele":
            decision = semantic_frame(
                primary_intent="CLARIFY",
                operation="clarify",
                agent=None,
                constraints=[],
                ambiguity={"present": True, "candidates": ["codex", "deepseek"]},
            )
        else:
            decision = semantic_frame(constraints=[])
        return SemanticPassResult(
            used=True,
            decision=decision,
            latency_ms=1.0,
            parse_valid=True,
            repair_used=False,
            cache_hit=False,
        )

    report = evaluate_explicit_agent_binding_ab(semantic_provider=provider)
    metrics = report["metrics"]

    assert report["ab_control"] == {
        "same_semantic_result_per_case": True,
        "semantic_calls": 24,
        "additional_inference_calls_B": 0,
        "schema_changed": False,
        "prompt_changed": False,
    }
    assert metrics["explicit_agent_detection_precision"] == 1.0
    assert metrics["explicit_agent_detection_recall"] == 1.0
    assert metrics["explicit_deepseek_route_accuracy"] == 1.0
    assert metrics["explicit_codex_route_accuracy"] == 1.0
    assert metrics["explicit_agent_preservation"] == 1.0
    assert metrics["false_agent_binding_rate"] == 0.0
    assert metrics["B"]["wrong_agent_rate"] == 0.0
    assert metrics["B"]["availability_intent_corruption_rate"] == 0.0
    assert metrics["fallback"] == 2


def test_no_second_inference_or_automatic_substitute_is_part_of_binding():
    assert QwenSemanticInterpreter.skipped().decision is None
    decision = AgentDecisionPolicy().decide(
        "mande isso para o DeepSeek",
        fixture_context={"active_project": "tern"},
        explicit_agent_binding=ExplicitAgentBinding("deepseek"),
    )
    assert decision.tools == ("delegate_to_deepseek",)
    assert "delegate_to_codex" not in decision.tools
