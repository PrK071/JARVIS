import argparse
import subprocess

from scripts.predictive_external_eval import evaluate_with_budget


def test_operational_timeout_is_not_abstention_or_project_execution(tmp_path, monkeypatch):
    invocations = []

    class Worker:
        returncode = None
        killed = False

        def __init__(self, command, **kwargs):
            invocations.append(command)

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(invocations[-1], timeout)
            return '', ''

        def kill(self):
            self.killed = True

    monkeypatch.setattr('scripts.predictive_external_eval.subprocess.Popen', Worker)
    args = argparse.Namespace(archive=tmp_path / 'archive', output=tmp_path / 'report.json',
                              corpus=tmp_path / 'corpus', snapshots=tmp_path / 'sources',
                              split='development', mode='live', case_timeout=1)
    value = evaluate_with_budget(args, [{'id': 'controlled'}])
    assert invocations[0][1:3] == ['-m', 'scripts.predictive_external_eval']
    assert 'pytest' not in invocations[0]
    assert value['operational_failures'][0]['status'] == 'NOT_EVALUATED_TIMEOUT'
    assert value['summary']['n_incomplete_defects'] == 1
    assert value['summary']['metrics']['false_abstention_rate']['denominator'] == 0
    assert value['summary']['metrics']['external_root_validity']['denominator'] == 0
