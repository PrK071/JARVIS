"""Sela runtime e evaluator antes da unica execucao do holdout externo."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    target = Path('tests/data/predictive/results/v13-runtime-freeze.json')
    if target.exists():
        raise ValueError('Freeze existente; nao sobrescrever.')
    corpus = Path('tests/data/predictive/external_v13/manifest.json')
    paths = [*Path('tern/orchestrator/predictive').glob('*.py'),
             Path('scripts/predictive_external_eval.py')]
    hashes = {path.as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for path in sorted(paths)}
    value = {'version': 13, 'frozen_at': datetime.now(timezone.utc).isoformat(),
             'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'corpus_seal': json.loads(corpus.read_text())['cases_sha256'],
             'runtime_sha256': hashes, 'holdout_live_executions_allowed': 1,
             'case_budget_seconds': 240,
             'holdout_thresholds': {'symbol_ledger_recall': .8, 'correct_candidate_recall': .75,
                                    'root_validity': .75, 'evidence_reference_validity': 1.,
                                    'unsupported_claim_rate': 0., 'forbidden_recommendations': 0},
             'no_post_holdout_tuning': True,
             'limitation': 'Purposive four-defect holdout cannot establish production readiness.'}
    target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
