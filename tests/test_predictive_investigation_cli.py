from test_predictive_investigation import InvestigationReasoner
from tern.orchestrator.cli import _print_predictive_report
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy


def test_human_cli_distinguishes_investigation_from_recommendation(tmp_path, capsys):
    (tmp_path / 'api.py').write_text('class Api:\n    def update(self, item):\n        self.current = item\n')
    report = PredictiveDecisionService(InvestigationReasoner(), path_policy=PathPolicy((tmp_path,))).predict(
        'Api.update() stores incorrect state', tmp_path)
    _print_predictive_report(report)
    output = capsys.readouterr().out
    assert 'Investigacao consultiva' in output
    assert 'nenhuma causa confirmada' in output
    assert 'Execucao nao autorizada' in output
    assert 'api.py::Api.update' in output
