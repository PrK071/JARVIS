"""Registra gates sem alterar o runtime ou misturar intervencoes oracle."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


RESULTS = Path('tests/data/predictive/results')


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def main():
    output = RESULTS / 'v13-stage-gate.json'
    if output.exists():
        raise ValueError('Nao sobrescrever gate historico.')
    freeze = read('v13-runtime-freeze.json')
    unchanged = all(hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
                    for path, digest in freeze['runtime_sha256'].items())
    if not unchanged:
        raise ValueError('Runtime divergiu do freeze antes do holdout.')
    dev = read('v13-development-final.json')
    holdout = read('v13-holdout-one-shot.json')
    structural = read('v13-synthetic-structural-final.json')
    live = read('v13-synthetic-live.json')
    audits = [read(name)['audits'][0] for name in (
        'v13-development-grounding-audit.json', 'v13-holdout-grounding-audit.json')]
    manifest = json.loads(Path('tests/data/predictive/external_v13/manifest.json').read_text())
    coverage_checks = {}
    decision_checks = {}
    for split, report in (('development', dev), ('holdout', holdout)):
        if report['corpus_seal'] != freeze['corpus_seal']:
            raise ValueError('Seal divergente.')
        layers = report['summary']['layer_coverage']['metrics']
        metrics = report['summary']['metrics']
        coverage_checks[split + '_complete'] = not report['operational_failures']
        coverage_checks[split + '_symbol_ledger'] = (layers['correct_symbol_in_ledger']['rate'] or 0) >= .8
        coverage_checks[split + '_root_candidate'] = (layers['correct_root_candidate_generated']['rate'] or 0) >= .75
        decision_checks[split + '_root_validity'] = (metrics['external_root_validity']['rate'] or 0) >= manifest['holdout_thresholds']['external_root_validity']
    safety = {key: sum(report['summary']['safety'][key] for report in (dev, holdout))
              for key in dev['summary']['safety']}
    safety.update(forbidden_recommendations=sum(audit['forbidden_recommendations'] for audit in audits),
                  unsupported_claims=sum(row['unsupported_claims'] for audit in audits for row in audit['rows']),
                  claims=sum(row['claims'] for audit in audits for row in audit['rows']),
                  valid_evidence_references=sum(row['valid_refs'] for audit in audits for row in audit['rows']),
                  evidence_references=sum(row['refs'] for audit in audits for row in audit['rows']))
    safety_passed = all(audit['passed'] for audit in audits) and not any(
        report['summary']['safety'][key] for report in (dev, holdout) for key in report['summary']['safety'])
    structural_passed = all(structural['metrics'][name]['value'] == 1 for name in (
        'oracle_root_candidate_recall', 'oracle_slot_identity_recall', 'oracle_scc_recall'))
    live_passed = all(live['layer_metrics'][name]['value'] == 1 for name in ('root_validity', 'repair_pair_validity'))
    coverage_ready = safety_passed and all(coverage_checks.values())
    value = {'version': 13, 'completed_at': datetime.now(timezone.utc).isoformat(),
             'runtime_unchanged_since_freeze': unchanged, 'corpus_seal': freeze['corpus_seal'],
             'holdout_live_executions': 1, 'post_holdout_tuning': False,
             'coverage_checks': coverage_checks, 'decision_checks': decision_checks,
             'gates': {'PREDICTIVE_SYNTHETIC_READY': structural_passed and live_passed,
                       'PREDICTIVE_EXTERNAL_COVERAGE_READY': coverage_ready,
                       'PREDICTIVE_EXTERNAL_DECISION_READY': coverage_ready and all(decision_checks.values()),
                       'SIMULATION_RUNTIME_READY': False},
             'safety': safety, 'development': dev['summary'], 'holdout': holdout['summary'],
             'limitations': ['Four independent holdout defects, not a population-level production claim.',
                             'Root-location matching and local AST edge coverage are proxies, not entailment proofs.',
                             'Operationally incomplete cases are excluded from rates and block promotion.',
                             'Oracle interventions excluded from real accuracy.',
                             'Timing observations include profiling and colocated evaluation load.',
                             'Off-oracle symbols are not automatically irrelevant; semantic noise not adjudicated.',
                             'Synthetic regression is a 172-case offline run and a 14-case live sample; not a new exhaustive live qualification.']}
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value['gates'], indent=2))


if __name__ == '__main__':
    main()
