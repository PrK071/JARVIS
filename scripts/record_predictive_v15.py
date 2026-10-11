"""Seals e gates independentes; mais hipoteses nao promovem diagnosticos."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path('tests/data/predictive/results')
CORPUS = Path('tests/data/predictive/external_v15')


def runtime_hashes():
    paths = [*Path('tern/orchestrator/predictive').glob('*.py'),
        Path('tern/orchestrator/project_intelligence_v2.py'), Path('tern/orchestrator/cli.py'),
        Path('scripts/predictive_external_eval.py'), Path('scripts/predictive_checkpoints.py')]
    return {str(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}


def checkpoint_audit():
    manifest = json.loads((RESULTS / 'v15-artifact-preservation-final.json').read_text())
    root = Path(manifest['destination']).resolve(strict=True)
    rows = []
    for item in manifest['checkpoints']:
        path = (root / item['relative_path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Checkpoint fora do backup.')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Hash do checkpoint divergente.')
        events = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        starts = [e for e in events if e['phase'] == 'qwen_request' and e['state'] == 'START']
        finishes = [e for e in events if e['phase'] == 'qwen_request' and e['state'] in {'END', 'ERROR'}]
        analysis = [e for e in events if e['phase'] == 'qwen_analysis' and e['state'] == 'START']
        rows.append({'journal': item['relative_path'],
            'requests_started': len(starts), 'requests_finished': len(finishes),
            'requests_unfinished': len(starts) - len(finishes),
            'pre_request_seconds': starts[0]['wall_seconds'] - analysis[0]['wall_seconds'] if starts and analysis else None,
            'last_checkpoint': events[-1] if events else None,
            'peak_sampled_working_set_bytes': max((e.get('working_set_bytes', 0) for e in events), default=0),
            'phase_seconds': {e['phase']: e['elapsed_seconds'] for e in events if e['state'] in {'END', 'ERROR'}}})
    return {'population': '6 external worker journals; completed and censored attempts',
        'rows': rows, 'requests_started': sum(r['requests_started'] for r in rows),
        'requests_finished': sum(r['requests_finished'] for r in rows),
        'requests_unfinished': sum(r['requests_unfinished'] for r in rows),
        'limitation': 'Request includes unknown model queue/generation; process memory is sampled, not a hard allocation cap.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--full-tests-passed', action='store_true')
    parser.add_argument('--output', type=Path, default=RESULTS / 'v15-stage-gate.json')
    parser.add_argument('--invalidate-external-qualification', action='store_true',
        help='Correcao independente pos-seal: manter holdout historico e bloquear promocao externa')
    args = parser.parse_args()
    freeze = RESULTS / 'v15-runtime-freeze.json'
    manifest = json.loads((CORPUS / 'manifest.json').read_text())
    if args.freeze:
        if freeze.exists():
            raise ValueError('Runtime ja congelado.')
        freeze.write_text(json.dumps({'version': 15, 'frozen_at': datetime.now(timezone.utc).isoformat(),
            'corpus_seal': manifest['cases_sha256'], 'runtime_source_sha256': runtime_hashes()}, indent=2)+'\n')
        return
    frozen = json.loads(freeze.read_text())
    changed_runtime = frozen['runtime_source_sha256'] != runtime_hashes()
    if changed_runtime and not args.invalidate_external_qualification:
        raise ValueError('Runtime mudou apos freeze; nao promover holdout.')
    if frozen['corpus_seal'] != manifest['cases_sha256']:
        raise ValueError('Corpus mudou apos freeze.')
    holdout = json.loads((RESULTS / 'v15-external-holdout-one-shot.json').read_text())
    synthetic = json.loads((RESULTS / 'v15-synthetic-structural-current.json').read_text())
    structural = synthetic['layer_metrics'] if 'layer_metrics' in synthetic else synthetic['metrics']
    summary = holdout['summary']
    investigation = summary['investigation']
    thresholds = manifest['holdout_thresholds']
    def passes(metric, threshold):
        return metric.get('denominator', 0) > 0 and metric.get('rate') is not None and metric['rate'] >= threshold
    complete = summary['n_incomplete_defects'] == 0 and summary['n_evaluable_defects'] == manifest['holdout']
    investigation_ready = complete and passes(investigation['hypothesis_location_precision'],
        thresholds['hypothesis_location_precision']) and passes(investigation['hypothesis_evidence_validity'],
        thresholds['hypothesis_evidence_validity']) and investigation['incorrect_hypothesis_promotion_rate']['numerator'] == 0
    coverage_ready = complete and passes(investigation['contract_availability'], thresholds['contract_availability']) and passes(
        summary['metrics']['correct_candidate_recall'], thresholds['correct_root_candidate_recall'])
    decision_ready = coverage_ready and passes(summary['metrics']['external_root_validity'],
        thresholds['external_root_validity']) and passes(summary['metrics']['repair_pair_validity'], thresholds['external_root_validity'])
    required_synthetic = {'oracle_root_candidate_recall', 'oracle_slot_identity_recall', 'oracle_scc_recall'}
    synthetic_ready = args.full_tests_passed and required_synthetic <= structural.keys() and all(
        structural[key]['numerator'] == structural[key]['denominator'] and structural[key]['denominator']
        for key in required_synthetic)
    if changed_runtime:
        investigation_ready = coverage_ready = decision_ready = False
    value = {'version': 15, 'corpus_seal': manifest['cases_sha256'],
        'PREDICTIVE_SYNTHETIC_READY': synthetic_ready,
        'INVESTIGATION_CAPABILITY_READY': investigation_ready,
        'PREDICTIVE_EXTERNAL_COVERAGE_READY': coverage_ready,
        'PREDICTIVE_EXTERNAL_DECISION_READY': decision_ready,
        'SIMULATION_RUNTIME_READY': False,
        'complete_external_holdout': complete, 'thresholds': thresholds,
        'safety': summary['safety'], 'full_tests_passed': args.full_tests_passed,
        'synthetic_regression_report': 'v15-synthetic-structural-current.json',
        'checkpoint_audit': checkpoint_audit(),
        'runtime_source_sha256': runtime_hashes(),
        'evaluated_runtime_source_sha256': frozen['runtime_source_sha256'],
        'external_qualification_invalidated': changed_runtime,
        'post_seal_correction': 'Builtin bool is a subtype of int; conservative contract check corrected by independent negative unit controls, not holdout tuning.' if changed_runtime else None,
        'limitations': ['Synthetic scope only; current stratified live regression is separately reported.',
            'Six independent external defects, three held out, cannot prove general population readiness.',
            'Hypothesis precision uses adjudicated source locations, not demonstrated causal defects.',
            'No automatic hypothesis promotion or repair recommendation; static promotion checks existing represented candidates.',
            'Docker unavailable; no host execution fallback.']}
    target = args.output
    if target.exists():
        raise ValueError('Nao sobrescrever stage gate.')
    target.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: v for k, v in value.items() if k.endswith('READY')}))


if __name__ == '__main__':
    main()
