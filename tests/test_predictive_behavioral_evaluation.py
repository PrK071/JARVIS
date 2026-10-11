import argparse
import ast
import json
from pathlib import Path

import pytest

from scripts.predictive_external_eval import EvaluationMemoryLimit, _communicate_bounded
from tern.orchestrator.predictive.coverage_evaluation import behavioral_coverage
from tern.orchestrator.predictive.external_evaluation import load_external_cases
from tern.orchestrator.predictive.operational import AnalysisTimer, PredictiveOperationalTimeout
from tern.orchestrator.predictive.grounding import _FactVisitor


def test_new_holdout_seal_disjoint_and_preregistered():
    cases = load_external_cases(Path('tests/data/predictive/external_v14'), split='all')
    old = {case['id'] for v in (12, 13) for case in load_external_cases(
        Path(f'tests/data/predictive/external_v{v}'), split='all')}
    assert len(cases) == 4
    assert not old.intersection(case['id'] for case in cases)
    assert len({c['repo'] for c in cases}) == 2
    assert sum(c['split'] == 'holdout' for c in cases) == 2
    manifest = json.loads(Path('tests/data/predictive/external_v14/manifest.json').read_text())
    assert manifest['holdout_thresholds']['candidate_generation_recall'] == .75


def test_memory_budget_never_executes_external_project(monkeypatch):
    class Worker:
        def communicate(self, **kwargs):
            raise AssertionError('Memoria excedida antes de aguardar.')
    monkeypatch.setattr('scripts.predictive_external_eval._working_set_bytes', lambda worker: 2_000_000)
    with pytest.raises(EvaluationMemoryLimit):
        _communicate_bounded(Worker(), ['python', '-m', 'scripts.predictive_external_eval'], 10, 1)


def test_cooperative_timeout_not_causal_failure(monkeypatch):
    # Inicio do timer, inicio do checkpoint, duracao e limite total.
    moments = iter([0, 0])
    monkeypatch.setattr('tern.orchestrator.predictive.operational.perf_counter', lambda: next(moments, 5))
    timer = AnalysisTimer(started=0, max_static_stage_seconds=1)
    with pytest.raises(PredictiveOperationalTimeout, match='NOT_EVALUATED_TIMEOUT'):
        timer.measure('ledger', lambda: ())
    assert timer.seconds['ledger'] == 5


@pytest.mark.parametrize('source', [
    'def example():\n    return "acao € 漢字"\n',
    'def example():\n    return (\n        compute(1),\n        compute(2)\n    )\n',
    'def example():\n    text = "form\fcontrol"\n    return text\n',
])
def test_cached_snippets_equal_python_ast_source_segments(source):
    visitor = _FactVisitor('api.py', source, set(range(1, 20)), [])
    for node in ast.walk(ast.parse(source)):
        if hasattr(node, 'end_lineno'):
            assert visitor._snippet(node) == (ast.get_source_segment(source, node) or '').strip()[:320]
