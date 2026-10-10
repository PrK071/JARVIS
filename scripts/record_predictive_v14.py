"""Freeze/gates reproduziveis, sem modificar engine ou resultados historicos."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RESULTS = Path('tests/data/predictive/results')


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def runtime_hashes():
    paths = list(Path('tern/orchestrator/predictive').rglob('*.py')) + [
        Path('tern/orchestrator/project_intelligence_v2.py'), Path('scripts/predictive_external_eval.py')]
    return {p.as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path('tests/data/predictive/external_v14/manifest.json').read_text())
    output = RESULTS / ('v14-runtime-freeze.json' if args.freeze else 'v14-stage-gate.json')
    if output.exists():
        raise ValueError('Resultado preservado; nao sobrescrever.')
    if args.freeze:
        value = {'version': 14, 'frozen_at': datetime.now(timezone.utc).isoformat(),
            'engine_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'runtime_sha256': runtime_hashes(), 'corpus_seal': manifest['cases_sha256'],
            'thresholds': manifest['holdout_thresholds'], 'case_timeout_seconds': 240,
            'case_working_set_monitor_mb': 1024,
            'limits': 'Stage checks cooperative (60s/static, 240s/total); evaluator worker hard wall timeout and sampled Windows RSS.',
            'coverage_requirements': ['Known adjudicated API path with eligible denominator',
                'Root candidate recall reaches preregistered threshold',
                'Contract representability and semantic noise independently adjudicated',
                'Grounding safety audited; no incomplete operational cases'],
            'no_post_holdout_tuning': True}
    else:
        freeze = read('v14-runtime-freeze.json')
        if freeze['runtime_sha256'] != runtime_hashes():
            raise ValueError('Engine mudou depois do freeze.')
        development = read('v14-external-development-live-final.json')
        holdout = read('v14-external-holdout-one-shot.json')
        structural = read('v14-synthetic-structural-final.json')
        live = read('v14-synthetic-live.json')
        if any(r['corpus_seal'] != freeze['corpus_seal'] for r in (development, holdout)):
            raise ValueError('Seal divergente.')
        audits = [read(n)['audits'][0] for n in (
            'v14-development-grounding-audit.json', 'v14-holdout-grounding-audit.json')]
        safety = {key: sum(r['summary']['safety'][key] for r in (development, holdout))
                  for key in development['summary']['safety']}
        safety.update(forbidden_recommendations=sum(a['forbidden_recommendations'] for a in audits),
            unsupported_claims=sum(row['unsupported_claims'] for a in audits for row in a['rows']),
            claims=sum(row['claims'] for a in audits for row in a['rows']),
            valid_evidence_references=sum(row['valid_refs'] for a in audits for row in a['rows']),
            evidence_references=sum(row['refs'] for a in audits for row in a['rows']))
        synthetic_ready = (all(structural['metrics'][key]['value'] == 1 for key in (
            'oracle_root_candidate_recall', 'oracle_slot_identity_recall', 'oracle_scc_recall'))
            and all(live['layer_metrics'][key]['value'] == 1 for key in ('root_validity', 'repair_pair_validity')))
        checks = {}
        for split, report in (('development', development), ('holdout', holdout)):
            layers = report['summary']['layer_coverage']
            checks[split+'_complete'] = not report['operational_failures']
            checks[split+'_candidate_recall'] = (layers['metrics']['correct_root_candidate_generated']['rate'] or 0) >= .75
            path_metric = layers['behavioral_metrics']['implementation_path_available']
            checks[split+'_api_implementation_path'] = bool(path_metric['denominator'] and (path_metric['rate'] or 0) >= .75)
        checks['independent_defect_relation_and_noise_adjudicated'] = False
        checks['grounding_audited'] = all(a['passed'] for a in audits)
        coverage_ready = all(checks.values())
        decision_ready = coverage_ready and all((r['summary']['metrics']['external_root_validity']['rate'] or 0) >= .75
                                                for r in (development, holdout))
        value = {'version': 14, 'completed_at': datetime.now(timezone.utc).isoformat(),
            'runtime_unchanged_since_freeze': True, 'corpus_seal': freeze['corpus_seal'],
            'holdout_live_executions': 1, 'post_holdout_tuning': False,
            'coverage_checks': checks, 'safety': safety,
            'gates': {'PREDICTIVE_SYNTHETIC_READY': synthetic_ready,
                'PREDICTIVE_EXTERNAL_COVERAGE_READY': coverage_ready,
                'PREDICTIVE_EXTERNAL_DECISION_READY': decision_ready,
                'SIMULATION_RUNTIME_READY': False},
            'development': development['summary'], 'holdout': holdout['summary'],
            'limitations': ['Four independent defects/two held out; no broad generalization claim.',
                'Operation reachability is not causal defect responsibility.',
                'Contract relation and external semantic noise not fully adjudicated.',
                'Oracle C paths not independently adjudicated; E selector not measured offline.',
                'Synthetic regression: 172 offline cases and 14 live cases, not exhaustive live v10 rerun.',
                'Working-set polling is sampled enforcement, not a hard memory allocation ceiling.',
                'Profiling/colocated tests affect measured latency.']}
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value.get('gates', {'runtime_frozen': True}), indent=2))


if __name__ == '__main__':
    main()
