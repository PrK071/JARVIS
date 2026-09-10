from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tern.orchestrator import cli
from tern.orchestrator.predictive.models import DecisionReport
from tern.orchestrator.security import PathPolicy


class FakeRuntime:
    def __init__(self, _settings):
        self.started = False

    def ensure_llama_server(self, _wait):
        self.started = True
        return {"healthy": True}


class FakeProjects:
    def __init__(self, root: Path):
        self.root = root
        self.policy = PathPolicy((root,))

    def active(self):
        return {"ok": True, "project": {"id": "repo", "root": str(self.root)}}


class FakeService:
    def __init__(self, _reasoner, *, path_policy):
        self.path_policy = path_policy

    def predict(self, problem, project_path):
        assert Path(project_path).is_dir()
        return DecisionReport(
            problem=problem,
            hypotheses=(),
            candidates=(),
            recommended_candidate_id=None,
            insufficient_evidence=True,
            recommendation_explanation="Nenhuma evidência forte.",
        )


def test_predict_parser_is_separate_from_agent_decision():
    args = cli.build_parser().parse_args(["predict", "TypeError", "--json"])

    assert args.command == "predict"
    assert args.problem == "TypeError"
    assert args.json_output is True


def test_predict_cli_is_read_only_and_emits_structured_report(monkeypatch, capsys, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(base_url="http://test", timeout=1))
    monkeypatch.setattr(cli, "RuntimeManager", FakeRuntime)
    monkeypatch.setattr(cli, "_predictive_project_registry", lambda _settings: FakeProjects(root))
    monkeypatch.setattr(cli, "PredictiveDecisionService", FakeService)
    monkeypatch.setattr(
        cli,
        "_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("predict must not construct a ToolRegistry")
        ),
    )

    assert cli.main(["predict", "TypeError", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)

    assert value["insufficient_evidence"] is True
    assert value["dry_run"] is True
    assert value["execution_authorized"] is False
    assert value["requires_approval"] is True
