from __future__ import annotations

import json
from types import SimpleNamespace

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.config import load_settings


TOOL_NAMES = (
    "resolve_project",
    "find_project_files",
    "filesystem_read_text",
    "review_codex_session",
    "get_codex_job_status",
    "delegate_to_codex",
    "steer_codex_job",
    "cancel_codex_job",
    "review_deepseek_session",
    "delegate_to_deepseek",
)


class Logger:
    def __init__(self):
        self.events = []

    def write_event(self, event, **values):
        self.events.append((event, values))


class Web:
    def begin_research(self, _text):
        return None

    def research_status(self):
        return {}


class Projects:
    def context(self):
        return {
            "active_project": {"id": "tern", "name": "Tern", "root": r"D:\tern"},
            "codex_thread_project": {"id": "tern"},
        }

    def context_text(self):
        return "Active project: tern\nProject root: D:\\tern"

    def projects(self):
        return [
            {"id": "tern", "name": "Tern", "root": r"D:\tern", "aliases": ["jarvis"]},
            {"id": "llama.cpp", "name": "llama.cpp", "root": r"D:\llama.cpp", "aliases": ["llama"]},
        ]


class Jobs:
    def __init__(self, running=False):
        self.running = running

    def list(self):
        if not self.running:
            return []
        return [{"job_id": "job-1", "status": "running"}]


class Codex:
    def __init__(self, running=False):
        self.jobs = Jobs(running)

    def claim_completed_results(self):
        return []

    def shared_project(self):
        return r"D:\tern"


class DeepSeek:
    def status(self, **_kwargs):
        return {"enabled": True, "configured": False, "active_session": "ds-1"}


class DryTools:
    def __init__(self, *, running=False):
        self.logger = Logger()
        self.web = Web()
        self.projects = Projects()
        self.codex = Codex(running)
        self.deepseek = DeepSeek()
        self.pending_actions = SimpleNamespace(pending=lambda: None)
        self.calls = []

    def specs(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in TOOL_NAMES
        ]

    def execute(self, name, arguments, **_kwargs):
        self.calls.append((name, arguments))
        results = {
            "find_project_files": {
                "ok": True,
                "root": r"D:\tern",
                "results": [{"path": "config.py"}],
            },
            "filesystem_read_text": {
                "ok": True,
                "path": arguments.get("path", r"D:\tern\config.py"),
                "content": "VOICE_RATE = 2",
            },
            "review_codex_session": {
                "ok": True,
                "thread_id": "thread-1",
                "turns_reviewed": 1,
                "summary_source": [{"result": "feito"}],
            },
            "get_codex_job_status": {"ok": True, "job_id": "job-1", "status": "completed"},
            "delegate_to_codex": {
                "ok": True,
                "job_id": "job-1",
                "thread_id": "thread-1",
                "turn_id": "turn-1",
                "status": "running",
            },
            "steer_codex_job": {"ok": True, "job_id": "job-1", "status": "running"},
            "cancel_codex_job": {"ok": True, "job_id": "job-1", "status": "cancelled"},
            "delegate_to_deepseek": {
                "ok": True,
                "session_id": "ds-1",
                "response": "parecer",
                "positive_recommendation": True,
            },
            "review_deepseek_session": {"ok": True, "session_id": "ds-1", "messages": []},
        }
        return results.get(name, {"ok": True})


class ToolThenAnswer:
    def __init__(self, tool=None, arguments=None):
        self.tool = tool
        self.arguments = arguments or {}
        self.turn = 0
        self.available = []

    def chat(self, _messages, **kwargs):
        self.turn += 1
        specs = kwargs.get("tools") or []
        self.available.append([item["function"]["name"] for item in specs])
        if self.turn == 1 and self.tool:
            assert self.tool in self.available[-1]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": self.tool,
                                        "arguments": json.dumps(self.arguments),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class SemanticScriptedClient:
    supports_structured_output = True

    def __init__(self, frame, tool_steps=()):
        self.frame = frame
        self.tool_steps = list(tool_steps)
        self.semantic_calls = 0
        self.action_calls = 0
        self.available = []

    def chat(self, _messages, **kwargs):
        if kwargs.get("response_format"):
            self.semantic_calls += 1
            assert kwargs.get("tools") is None
            return {"choices": [{"message": {"content": json.dumps(self.frame)}}]}
        specs = kwargs.get("tools") or []
        names = [item["function"]["name"] for item in specs]
        self.available.append(names)
        if self.action_calls < len(self.tool_steps):
            name, arguments = self.tool_steps[self.action_calls]
            self.action_calls += 1
            assert name in names
            return {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"semantic-call-{self.action_calls}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }],
                }}]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def semantic_frame(**changes):
    value = {
        "speech_act": "QUESTION",
        "primary_intent": "ANSWER_DIRECTLY",
        "operation": "answer",
        "execution_requested": False,
        "agent": "qwen",
        "target": {"type": "none", "reference": None},
        "constraints": [],
        "followup_type": "NEW_REQUEST",
        "continuation": False,
        "compound_plan": [],
        "ambiguity": {"present": False, "candidates": []},
        "confidence": 0.98,
    }
    value.update(changes)
    return value


def supervisor(tools, client):
    return Supervisor(
        load_settings(
            {
                "MODEL_MAX_TOOL_CALLS": "6",
                "AGENT_DECISION_FAST_PATH": "false",
            }
        ),
        client,
        tools,
    )


def test_e2e_codex_status_is_one_read_and_zero_new_turns():
    tools = DryTools(running=True)
    result = supervisor(tools, ToolThenAnswer("get_codex_job_status", {"latest": True})).run("o codex terminou?")
    assert result["ok"] and result["tool_calls"] == 1
    assert [name for name, _ in tools.calls] == ["get_codex_job_status"]


def test_e2e_codex_history_is_one_read_and_zero_new_turns():
    tools = DryTools()
    result = supervisor(tools, ToolThenAnswer("review_codex_session", {"turn_limit": 1})).run("oq o codex fez por ultimo?")
    assert result["ok"] and result["tool_calls"] == 1
    assert [name for name, _ in tools.calls] == ["review_codex_session"]


def test_e2e_project_mutation_delegates_once_to_tern():
    tools = DryTools()
    arguments = {"task": "corrigir config", "project_path": r"D:\tern", "wait": False}
    result = supervisor(tools, ToolThenAnswer("delegate_to_codex", arguments)).run("corrige o bug da config")
    assert result["ok"] and result["tool_calls"] == 1
    assert tools.calls == [("delegate_to_codex", arguments)]


def test_e2e_explicit_deepseek_has_zero_codex_calls():
    tools = DryTools()
    result = supervisor(tools, ToolThenAnswer("delegate_to_deepseek", {"task": "avalie"})).run("pergunta pro deepseek oq ele acha dessa solucao")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["delegate_to_deepseek"]


def test_e2e_direct_answer_has_zero_tools():
    tools = DryTools()
    client = ToolThenAnswer()
    result = supervisor(tools, client).run("essa solucao faz sentido?")
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == []
    assert client.available == [[]]


def test_e2e_find_read_then_direct_reuses_focus_without_second_search():
    tools = DryTools()
    agent = supervisor(tools, ToolThenAnswer("find_project_files", {"project_id": "tern", "query": "config voz"}))
    assert agent.run("onde ta a config da voz?")["ok"]
    agent.client = ToolThenAnswer("filesystem_read_text", {"path": r"D:\tern\config.py", "max_bytes": 4096})
    assert agent.run("abre ela")["ok"]
    agent.client = ToolThenAnswer()
    assert agent.run("agora explica")["ok"]
    assert [name for name, _ in tools.calls] == ["find_project_files", "filesystem_read_text"]


def test_e2e_active_job_followup_steers_same_job():
    tools = DryTools(running=True)
    agent = supervisor(tools, ToolThenAnswer("steer_codex_job", {"job_id": "job-1", "instruction": "inclua apenas falhas"}))
    agent.decision_policy.focus.focused_agent = "codex"
    result = agent.run("fala pra ele incluir apenas falhas")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["steer_codex_job"]


def test_e2e_active_job_followup_cancels_same_job():
    tools = DryTools(running=True)
    agent = supervisor(tools, ToolThenAnswer("cancel_codex_job", {"job_id": "job-1"}))
    agent.decision_policy.focus.focused_agent = "codex"
    result = agent.run("para ele")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["cancel_codex_job"]


def test_e2e_read_only_fast_path_executes_before_qwen(tmp_path):
    tools = DryTools(running=True)
    client = ToolThenAnswer()
    agent = Supervisor(
        load_settings(
            {
                "MODEL_STATE_DIR": str(tmp_path),
                "AGENT_DECISION_FAST_PATH": "true",
            }
        ),
        client,
        tools,
    )
    result = agent.run("o codex terminou?")
    assert result["ok"]
    assert tools.calls == [
        ("get_codex_job_status", {"job_id": "job-1", "latest": False})
    ]
    assert client.available == [[]]
    assert result["decision"]["fast_path"] is True
    assert result["timing"]["first_tool_ms"] < result["timing"]["response_ms"]


def test_e2e_shadow_records_final_outcome_without_changing_answer(tmp_path):
    tools = DryTools()
    agent = Supervisor(
        load_settings(
            {
                "MODEL_STATE_DIR": str(tmp_path),
                "AGENT_DECISION_SHADOW": "true",
                "AGENT_DECISION_FAST_PATH": "false",
            }
        ),
        ToolThenAnswer(),
        tools,
    )
    result = agent.run("essa solucao faz sentido?")
    assert result["answer"] == "ok"
    values = [
        json.loads(line)
        for line in (tmp_path / "agent-decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert values[-1]["outcome"] == "direct_answer"
    assert values[-1]["actual_tools"] == []


def test_e2e_question_about_cancel_explains_without_cancelling():
    tools = DryTools(running=True)
    agent = supervisor(tools, ToolThenAnswer())
    agent.decision_policy.focus.focused_agent = "codex"
    result = agent.run("como eu cancelo o Codex?")
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == []


def test_e2e_explicit_cancel_still_cancels_active_job():
    tools = DryTools(running=True)
    result = supervisor(
        tools,
        ToolThenAnswer("cancel_codex_job", {"job_id": "job-1"}),
    ).run("cancela o Codex")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["cancel_codex_job"]


def test_e2e_negated_cancel_routes_only_to_status():
    tools = DryTools(running=True)
    agent = supervisor(
        tools,
        ToolThenAnswer("get_codex_job_status", {"job_id": "job-1"}),
    )
    agent.decision_policy.focus.focused_agent = "codex"
    result = agent.run("não cancela, só vê se terminou")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["get_codex_job_status"]


def test_e2e_deepseek_capability_question_does_not_call_api():
    tools = DryTools()
    result = supervisor(tools, ToolThenAnswer()).run("o DeepSeek conseguiria analisar isso?")
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == []


def test_e2e_explicit_deepseek_question_calls_only_deepseek():
    tools = DryTools()
    result = supervisor(
        tools,
        ToolThenAnswer("delegate_to_deepseek", {"task": "analisar capacidade"}),
    ).run("pergunta ao DeepSeek se ele conseguiria analisar isso")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["delegate_to_deepseek"]


def test_e2e_negated_deepseek_override_answers_directly():
    tools = DryTools()
    result = supervisor(tools, ToolThenAnswer()).run(
        "não pergunta pro DeepSeek, quero sua opinião"
    )
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == []


def test_e2e_codex_review_then_opinion_reuses_result():
    tools = DryTools()
    agent = supervisor(
        tools,
        ToolThenAnswer("review_codex_session", {"turn_limit": 1}),
    )
    assert agent.run("mostra o que o Codex fez e me diz o que você acha")["ok"]
    agent.client = ToolThenAnswer()
    assert agent.run("e o que você mudaria?")["ok"]
    assert [name for name, _ in tools.calls] == ["review_codex_session"]


def test_e2e_negative_codex_instruction_steers_instead_of_delegating():
    tools = DryTools(running=True)
    agent = supervisor(
        tools,
        ToolThenAnswer(
            "steer_codex_job",
            {"job_id": "job-1", "instruction": "não mexer no TTS"},
        ),
    )
    agent.decision_policy.focus.focused_agent = "codex"
    assert agent.run("fala pra ele não mexer no TTS")["ok"]
    assert [name for name, _ in tools.calls] == ["steer_codex_job"]


def test_e2e_question_about_steering_does_not_steer():
    tools = DryTools(running=True)
    agent = supervisor(tools, ToolThenAnswer())
    agent.decision_policy.focus.focused_agent = "codex"
    result = agent.run("como eu falo pro Codex não mexer no TTS?")
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == []


def test_e2e_semantic_first_question_about_cancel_never_cancels():
    tools = DryTools(running=True)
    client = SemanticScriptedClient(semantic_frame())
    result = supervisor(tools, client).run("como faço o Codex parar?")
    assert result["ok"] and result["tool_calls"] == 0
    assert client.semantic_calls == 1 and tools.calls == []


def test_e2e_semantic_first_command_cancels_after_validation():
    tools = DryTools(running=True)
    frame = semantic_frame(
        speech_act="COMMAND",
        primary_intent="CODEX_CANCEL",
        operation="cancel",
        execution_requested=True,
        agent="codex",
        target={"type": "codex_job", "reference": "latest_codex_job"},
    )
    client = SemanticScriptedClient(frame, [("cancel_codex_job", {"job_id": "job-1"})])
    result = supervisor(tools, client).run("faz o Codex parar")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["cancel_codex_job"]


def test_e2e_semantic_first_negated_cancel_exposes_status_only():
    tools = DryTools(running=True)
    frame = semantic_frame(
        speech_act="STATUS_QUERY",
        primary_intent="CODEX_STATUS",
        operation="status",
        agent="codex",
        target={"type": "codex_job", "reference": "latest_codex_job"},
        constraints=["FORBID_CANCEL", "FORBID_NEW_TURN"],
    )
    client = SemanticScriptedClient(frame, [("get_codex_job_status", {"job_id": "job-1"})])
    result = supervisor(tools, client).run("não faz o Codex parar, só me diz se terminou")
    assert result["ok"]
    assert client.available[0] == ["get_codex_job_status"]
    assert [name for name, _ in tools.calls] == ["get_codex_job_status"]


def test_e2e_semantic_first_answer_self_filters_delegation():
    tools = DryTools()
    frame = semantic_frame(constraints=["FORBID_DEEPSEEK", "ANSWER_SELF"])
    client = SemanticScriptedClient(frame)
    result = supervisor(tools, client).run("não vê com o DeepSeek, responde você")
    assert result["ok"] and result["tool_calls"] == 0
    assert tools.calls == [] and client.available == [[]]


def test_e2e_semantic_first_conditional_compound_plan_is_ordered():
    tools = DryTools()
    plan = [
        {
            "intent": "DEEPSEEK_DELEGATE", "operation": "delegate", "agent": "deepseek",
            "target_type": "task", "target_reference": "user_mentioned_target", "condition": None,
        },
        {
            "intent": "CODEX_DELEGATE", "operation": "delegate", "agent": "codex",
            "target_type": "task", "target_reference": "deepseek_recommendation",
            "condition": "positive_recommendation",
        },
    ]
    frame = semantic_frame(
        speech_act="COMMAND",
        primary_intent="DEEPSEEK_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="deepseek",
        constraints=["ORDERED"],
        compound_plan=plan,
    )
    client = SemanticScriptedClient(
        frame,
        [
            ("delegate_to_deepseek", {"task": "avalie"}),
            ("delegate_to_codex", {"task": "revise", "project_path": r"D:\tern", "wait": False}),
        ],
    )
    result = supervisor(tools, client).run(
        "pergunta ao DeepSeek e se ele concordar manda o Codex revisar"
    )
    assert result["ok"]
    assert client.available[:2] == [["delegate_to_deepseek"], ["delegate_to_codex"]]
    assert [name for name, _ in tools.calls] == ["delegate_to_deepseek", "delegate_to_codex"]


def test_e2e_conditional_plan_never_runs_unverified_second_step():
    tools = DryTools()
    original_execute = tools.execute

    def execute(name, arguments, **kwargs):
        result = original_execute(name, arguments, **kwargs)
        if name == "delegate_to_deepseek":
            result.pop("positive_recommendation")
        return result

    tools.execute = execute
    plan = [
        {
            "intent": "DEEPSEEK_DELEGATE", "operation": "delegate", "agent": "deepseek",
            "target_type": "task", "target_reference": "user_mentioned_target", "condition": None,
        },
        {
            "intent": "CODEX_DELEGATE", "operation": "delegate", "agent": "codex",
            "target_type": "task", "target_reference": "deepseek_recommendation",
            "condition": "positive_recommendation",
        },
    ]
    frame = semantic_frame(
        speech_act="COMMAND",
        primary_intent="DEEPSEEK_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="deepseek",
        constraints=["ORDERED"],
        compound_plan=plan,
    )
    client = SemanticScriptedClient(
        frame,
        [
            ("delegate_to_deepseek", {"task": "avalie"}),
        ],
    )
    result = supervisor(tools, client).run("pergunta ao DeepSeek e se ele concordar manda o Codex revisar")
    assert result["ok"]
    assert [name for name, _ in tools.calls] == ["delegate_to_deepseek"]
    assert client.available == [["delegate_to_deepseek"], []]
