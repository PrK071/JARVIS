from __future__ import annotations

from tern.orchestrator.cli import build_parser
from tern.orchestrator.text import TextSession


class Console:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.values: list[str] = []

    def write(self, value=""):
        self.values.append(value)

    def read(self, prompt=""):
        self.values.append(prompt)
        return next(self.answers)


class Supervisor:
    def __init__(self):
        self.prompts: list[str] = []

    def run(self, prompt, *, event_callback):
        self.prompts.append(prompt)
        event_callback("tool_start", {"name": "resolve_project"})
        return {"ok": True, "answer": f"resposta: {prompt}", "tool_calls": 1}


def test_text_command_is_registered():
    parser = build_parser()
    assert parser.parse_args(["text"]).command == "text"
    assert parser.parse_args(["text", "--once"]).once is True


def test_typed_session_sends_prompt_and_prints_answer():
    console = Console(["Onde esta config.py?", "/sair"])
    supervisor = Supervisor()
    result = TextSession(supervisor, console=console).run()
    assert result["ok"] and result["interactions"] == 1
    assert supervisor.prompts == ["Onde esta config.py?"]
    assert "resposta: Onde esta config.py?" in console.values
    assert "[assistente] executando ferramenta: resolve_project" in console.values


def test_typed_session_ignores_blank_input():
    console = Console(["   ", "oi", "/sair"])
    supervisor = Supervisor()
    TextSession(supervisor, console=console).run()
    assert supervisor.prompts == ["oi"]


def test_typed_session_once_returns_after_one_prompt():
    console = Console(["teste"])
    supervisor = Supervisor()
    result = TextSession(supervisor, console=console).run(once=True)
    assert result["interactions"] == 1
    assert supervisor.prompts == ["teste"]
