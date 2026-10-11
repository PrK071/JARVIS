"""Amostra live controlada: mede investigacao, nunca acerto de diagnostico."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.evaluation import CountingReasoner
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Nao sobrescrever historico.')
    families = (
        ('state', 'class Api:\n    def update(self, item):\n        self.current = item\n',
         'Api.update() stores unexpected state\nExpected: supplied item\nObserved: different state', 'Api.update'),
        ('yield', 'def read():\n    yield 1\n',
         'read() yields incorrect output\nExpected: two items\nObserved: one item', 'read'),
        ('no_anchor', 'def unrelated():\n    return None\n',
         'Unspecified operation returned an incorrect result', None),
    )
    rows, decisions = [], []
    for name, source, problem, expected in families:
        root = args.workspace.resolve() / name
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'api.py'
        path.write_text(source, encoding='utf-8')
        before = path.read_bytes()
        reasoner = CountingReasoner(LlamaClient('http://127.0.0.1:8080', 180))
        start = perf_counter()
        report = PredictiveDecisionService(reasoner, path_policy=PathPolicy((root,))).predict(problem, root)
        wall = perf_counter()-start
        assert path.read_bytes() == before
        assert not report.candidates and report.recommended_candidate_id is None
        assert not report.execution_authorized and report.requires_approval
        hypotheses = report.investigable_hypotheses
        correct = bool(hypotheses) and all(h.symbol == expected for h in hypotheses) if expected else not hypotheses
        row = {'family': name, 'expected': 'INVESTIGATION' if expected else 'ABSTENTION',
            'location_or_abstention_correct': correct, 'hypothesis_count': len(hypotheses),
            'confirmed_roots': len(report.root_cause_candidates), 'recommendations': 0,
            'qwen_requests': reasoner.requests, 'request_ms': reasoner.request_ms,
            'wall_seconds': wall, 'phase_seconds': dict(report.phase_seconds),
            'filesystem_mutations': 0, 'tool_dispatches': reasoner.tool_dispatches,
            'authority_grants': 0, 'execution_authorized': 0}
        rows.append(row)
        decisions.append(report.as_dict())
        print(json.dumps(row), flush=True)
    raw = json.dumps(decisions, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    args.archive.mkdir(parents=True, exist_ok=True)
    archive = args.archive / (digest + '.json.gz')
    archive.write_bytes(gzip.compress(raw, mtime=0))
    args.output.write_text(json.dumps({'version': 15, 'population': '3 controlled local families, not external root accuracy',
        'rows': rows, 'raw_trace_sha256': digest, 'raw_trace_archive': str(archive)}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
